"""Matching Engine de ativo único com Price-Time Priority (FIFO).

Premissas de design:
    * Preços: sempre ``decimal.Decimal`` (nunca ``float``).
    * Prioridade temporal: ``seq`` inteiro e determinístico
      (``itertools.count``), atribuído pela engine. Menor ``seq`` = mais antiga.
    * Quantidade: ``leaves_qty`` (restante executável) é a fonte de verdade para
      execução e alteração, evitando ressuscitar liquidez já executada.
    * Cada lado do livro (``BookSide``) mantém ``levels`` (preço -> fila FIFO),
      ``prices`` (vetor ordenado via ``bisect``; melhor bid em ``[-1]``, melhor
      ask em ``[0]``) e ``pegs`` (fila isolada de ordens pegged).
    * Peg virtual: o peg nunca é reprecificado fisicamente. Seu preço é derivado
      em ``head()`` do melhor preço de ordens reais do mesmo lado; sem
      referência, fica dormente.
    * Invariante: todo preço em ``prices`` possui fila não vazia em ``levels``.
    * Execução: ao preço da ordem passiva, capturado antes da sua remoção.
      Execuções consecutivas no mesmo preço são consolidadas em uma única linha
      de trade (inferido do exemplo do enunciado).
    * Market não repousa (o saldo é descartado). Limit que cruza executa e o
      saldo repousa. Peg é passivo e nunca agride.
    * Cancelamento e alteração atuam apenas sobre ordens em repouso e nunca
      agridem o livro oposto.
"""

from __future__ import annotations

import itertools
from bisect import bisect_left, insort
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum


class Side(str, Enum):
    """Lado da ordem."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Tipo da ordem."""

    LIMIT = "limit"
    MARKET = "market"
    PEG = "peg"


@dataclass(slots=True, eq=False)
class Order:
    """Ordem do livro.

    ``eq=False`` mantém a comparação por identidade (a ordem é uma entidade,
    não um valor) e preserva o ``__hash__`` padrão.

    Attributes:
        id: Identificador único da ordem (chave em ``index`` e nas filas).
        side: Lado da ordem (compra ou venda).
        type: Tipo da ordem (limit, market ou peg).
        qty: Quantidade total da ordem (executada + restante).
        seq: Número de sequência que define a prioridade temporal. É atribuído
            pela engine; ``0`` significa "ainda não atribuído".
        price: Preço limite. ``None`` para market e peg.
        leaves_qty: Quantidade restante executável. Inicia igual a ``qty``.
    """

    id: str
    side: Side
    type: OrderType
    qty: int
    seq: int = 0
    price: Decimal | None = None
    leaves_qty: int = field(init=False)

    def __post_init__(self) -> None:
        self.leaves_qty = self.qty

    @property
    def is_peg(self) -> bool:
        """Indica se a ordem é do tipo Pegged."""
        return self.type is OrderType.PEG


class BookSide:
    """Um lado (compra ou venda) do livro de ofertas.

    Estruturas:
        levels: Mapa preço -> fila FIFO de ordens limit reais naquele preço.
        prices: Vetor de preços ativos, ordenado de forma crescente (mantido
            via ``bisect``). O melhor bid fica em ``[-1]`` e o melhor ask em
            ``[0]``.
        pegs: Fila FIFO das ordens Pegged deste lado, separada dos níveis.

    Invariante:
        todo preço em ``prices`` possui uma fila **não vazia** em ``levels``.
    """

    __slots__ = ("side", "levels", "prices", "pegs")

    def __init__(self, side: Side) -> None:
        self.side: Side = side
        self.levels: dict[Decimal, OrderedDict[str, Order]] = {}
        self.prices: list[Decimal] = []
        self.pegs: OrderedDict[str, Order] = OrderedDict()

    def best_price(self) -> Decimal | None:
        """Retorna o melhor preço de ordens reais, ou ``None`` se não houver.

        Pegs não participam: a referência de um peg é sempre o melhor preço
        das ordens *não pegged* do mesmo lado.
        """
        if not self.prices:
            return None
        return self.prices[-1] if self.side is Side.BUY else self.prices[0]

    def head(self) -> tuple[Order, Decimal] | None:
        """Retorna a próxima ordem na prioridade e seu preço de execução.

        Compara a ordem real mais antiga do melhor nível com o peg mais antigo.
        Vence o menor ``seq``; o peg executa ao preço de referência atual.
        Sem referência (lado sem ordens reais), os pegs ficam dormentes e o
        retorno é ``None``.

        O preço é devolvido junto da ordem para que quem executa o capture
        *antes* de remover a ordem passiva (o nível pode esvaziar).

        Raises:
            RuntimeError: se a invariante de níveis não vazios for violada.
        """
        melhor = self.best_price()
        if melhor is None:
            return None

        real = next(iter(self.levels[melhor].values()), None)
        if real is None:
            raise RuntimeError(f"Invariante violada: nível {melhor} vazio em prices")

        peg = next(iter(self.pegs.values()), None)
        if peg is not None and peg.seq < real.seq:
            return peg, melhor
        return real, melhor

    def add_order(self, order: Order) -> None:
        """Insere uma ordem limit real ao final da fila do seu nível de preço.

        Se o preço ainda não existir, cria a fila do nível e registra o preço
        em ``prices`` mantendo a ordenação crescente (``bisect.insort``).
        A ordem entra no fim da fila, preservando FIFO (menor ``seq`` primeiro).

        Args:
            order: Ordem do tipo LIMIT, com preço, pertencente a este lado.

        Raises:
            ValueError: se a ordem não for LIMIT, não tiver preço ou for do
                lado oposto ao deste ``BookSide``.
        """
        if order.type is not OrderType.LIMIT or order.price is None:
            raise ValueError("add_order aceita apenas ordens LIMIT com preço")
        if order.side is not self.side:
            raise ValueError("ordem pertence ao lado oposto do livro")

        fila = self.levels.get(order.price)
        if fila is None:
            fila = self.levels[order.price] = OrderedDict()
            insort(self.prices, order.price)
        fila[order.id] = order

    def remove_order(self, order: Order) -> None:
        """Remove uma ordem limit real do seu nível de preço.

        Invariante: se a fila ficar vazia, o nível é apagado de ``levels`` e o
        preço é removido de ``prices`` na mesma operação.

        A localização do preço usa busca binária (O(log L)); a remoção do
        vetor em si continua O(L) por deslocamento de memória.

        Args:
            order: Ordem LIMIT atualmente presente neste lado do livro.

        Raises:
            KeyError: se o nível ou a ordem não existirem neste lado.
        """
        preco = order.price
        fila = self.levels[preco]  # KeyError se o nível não existir
        del fila[order.id]  # KeyError se a ordem não estiver no nível

        if not fila:
            del self.levels[preco]
            del self.prices[bisect_left(self.prices, preco)]

    def add_peg(self, order: Order) -> None:
        """Insere uma ordem pegged ao final da fila ``pegs``.

        Não toca em ``levels`` nem em ``prices``: o peg nunca é reprecificado
        fisicamente, seu preço é derivado em ``head()``.

        Args:
            order: Ordem do tipo PEG pertencente a este lado.

        Raises:
            ValueError: se a ordem não for PEG ou for do lado oposto.
        """
        if not order.is_peg:
            raise ValueError("add_peg aceita apenas ordens PEG")
        if order.side is not self.side:
            raise ValueError("ordem pertence ao lado oposto do livro")
        self.pegs[order.id] = order

    def remove_peg(self, order: Order) -> None:
        """Remove uma ordem pegged da fila ``pegs``.

        Args:
            order: Ordem PEG atualmente presente neste lado do livro.

        Raises:
            KeyError: se a ordem não estiver em ``pegs``.
        """
        del self.pegs[order.id]


class MatchingEngine:
    """Motor de matching de um único ativo.

    Attributes:
        bids: Lado da compra do livro.
        asks: Lado da venda do livro.
        index: Mapa global ``id -> Order`` das ordens que repousam no livro
            (permite localizar uma ordem em O(1)).
        seq_gen: Emissor determinístico de ``seq`` (prioridade temporal).
        trades: Registro em memória dos trades, no formato
            ``"Trade, price: <preço>, qty: <quantidade>"``.
    """

    def __init__(self) -> None:
        self.bids: BookSide = BookSide(Side.BUY)
        self.asks: BookSide = BookSide(Side.SELL)
        self.index: dict[str, Order] = {}
        self.seq_gen = itertools.count(1)
        self.trades: list[str] = []

    # ------------------------------------------------------------------
    # Roteador
    # ------------------------------------------------------------------

    def process_order(self, incoming: Order) -> None:
        """Processa uma nova ordem: atribui ``seq``, casa e, se couber, repousa.

        Fluxo:
            1. Atribui o ``seq`` (prioridade temporal).
            2. Ordens MARKET e LIMIT agridem o lado oposto (``_match``).
               Ordens PEG são passivas por definição e pulam esta etapa.
            3. O saldo de LIMIT/PEG repousa no livro e entra no ``index``.
               O saldo de MARKET evapora (não há preço para repousar).

        Args:
            incoming: Ordem nova, ainda fora do ``index``. O campo ``seq`` é
                sobrescrito pela engine.

        Raises:
            ValueError: se o ``id`` já estiver em uso por uma ordem no livro.
        """
        if incoming.id in self.index:
            raise ValueError(f"id de ordem duplicado: {incoming.id}")

        incoming.seq = next(self.seq_gen)

        # Peg não tem preço próprio (price=None): nunca agride, só repousa.
        if not incoming.is_peg:
            self._match(incoming)

        if incoming.leaves_qty > 0 and incoming.type is not OrderType.MARKET:
            lado = self._lado(incoming)
            if incoming.is_peg:
                lado.add_peg(incoming)
            else:
                lado.add_order(incoming)
            self.index[incoming.id] = incoming

    # ------------------------------------------------------------------
    # Cancelamento e alteração
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> None:
        """Cancela uma ordem em repouso, retirando-a do livro e do ``index``.

        Args:
            order_id: Identificador da ordem a cancelar.

        Raises:
            KeyError: se a ordem não existir em repouso (inexistente, já
                executada, já cancelada ou market).
        """
        order = self.index[order_id]  # O(1); KeyError se não existir
        lado = self._lado(order)
        if order.is_peg:
            lado.remove_peg(order)
        else:
            lado.remove_order(order)
        del self.index[order_id]

    def amend_order(
        self, order_id: str, new_qty: int, new_price: Decimal | None = None
    ) -> None:
        """Altera preço e/ou quantidade de uma ordem em repouso.

        ``new_qty`` é a nova quantidade *restante* (``leaves_qty``); ``qty``
        é recomposta como ``executado + new_qty``. Comparar contra o restante,
        e não contra a quantidade original, impede ressuscitar liquidez de
        ordens parcialmente executadas.

        Regras de prioridade:
            * Mudança de preço ou aumento de quantidade: perde a prioridade.
              A ordem sai do livro, é atualizada, recebe novo ``seq`` e é
              reinserida no fim da fila do nível adequado.
            * Redução estrita de quantidade no mesmo preço: mantém ``seq`` e
              a posição na fila (atualização no próprio lugar).
            * Nenhuma mudança efetiva: no-op.

        A alteração nunca agride o livro oposto: um novo preço que cruzaria o
        spread é rejeitado. Toda validação ocorre antes de qualquer mutação,
        então uma chamada que falha deixa o livro intacto.

        Args:
            order_id: Identificador da ordem em repouso.
            new_qty: Nova quantidade restante (deve ser > 0).
            new_price: Novo preço. ``None`` mantém o preço atual. Não se
                aplica a ordens PEG.

        Raises:
            KeyError: se a ordem não existir em repouso (market nunca repousa).
            ValueError: se ``new_qty <= 0``, se informar preço para um PEG ou
                se o novo preço cruzaria o spread.
        """
        order = self.index[order_id]  # O(1); KeyError se não existir

        if new_qty <= 0:
            raise ValueError("new_qty deve ser positiva; use cancel_order")

        if order.is_peg:
            if new_price is not None:
                raise ValueError("ordem PEG não possui preço próprio")
            preco_alvo = None
            mudou_preco = False
        else:
            preco_alvo = order.price if new_price is None else new_price
            mudou_preco = preco_alvo != order.price

        if not mudou_preco and new_qty == order.leaves_qty:
            return  # nada a alterar

        if mudou_preco and self._cruzaria(order, preco_alvo):
            raise ValueError("alteração cruzaria o spread; amend nunca agride")

        executado = order.qty - order.leaves_qty

        if not mudou_preco and new_qty < order.leaves_qty:
            # Redução estrita: atualiza no lugar, preservando seq e posição.
            order.qty = executado + new_qty
            order.leaves_qty = new_qty
            return

        # Perda de prioridade: remove (usa o preço antigo para localizar),
        # atualiza, renova o seq e reinsere no fim da fila.
        lado = self._lado(order)
        if order.is_peg:
            lado.remove_peg(order)
        else:
            lado.remove_order(order)

        order.price = preco_alvo
        order.qty = executado + new_qty
        order.leaves_qty = new_qty
        order.seq = next(self.seq_gen)

        if order.is_peg:
            lado.add_peg(order)
        else:
            lado.add_order(order)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match(self, incoming: Order) -> None:
        """Loop de agressão da ordem ``incoming`` contra o lado oposto.

        O preço de execução vem de ``head()`` e é capturado *antes* de a ordem
        passiva ser removida (o nível pode esvaziar). Execuções consecutivas
        no mesmo preço são consolidadas em uma única linha de trade, conforme
        inferido do exemplo do enunciado (100 + 50 @ 20 => ``qty: 150``).

        Args:
            incoming: Ordem MARKET ou LIMIT já com ``seq`` atribuído.
        """
        oposto = self.asks if incoming.side is Side.BUY else self.bids
        preco_acum: Decimal | None = None
        qtd_acum = 0

        while incoming.leaves_qty > 0:
            topo = oposto.head()
            if topo is None:  # livro oposto esgotado (ou só pegs dormentes)
                break
            passiva, preco = topo  # preço capturado antes de qualquer remoção

            if incoming.type is OrderType.LIMIT and not self._favoravel(
                incoming.side, incoming.price, preco
            ):
                break

            qtd = min(incoming.leaves_qty, passiva.leaves_qty)
            incoming.leaves_qty -= qtd
            passiva.leaves_qty -= qtd

            if passiva.leaves_qty == 0:
                if passiva.is_peg:
                    oposto.remove_peg(passiva)
                else:
                    oposto.remove_order(passiva)
                del self.index[passiva.id]

            if preco != preco_acum:  # mudou o preço: fecha a linha anterior, se for a primeira, ignora
                self._registrar_trade(preco_acum, qtd_acum)
                preco_acum, qtd_acum = preco, 0
            qtd_acum += qtd

        self._registrar_trade(preco_acum, qtd_acum)

    # ------------------------------------------------------------------
    # Auxiliares
    # ------------------------------------------------------------------

    def _lado(self, order: Order) -> BookSide:
        """Retorna o ``BookSide`` onde a ordem repousa."""
        return self.bids if order.side is Side.BUY else self.asks

    """Independência de estado: avalia os argumentos sem consultar ou alterar variáveis do motor."""
    @staticmethod
    def _favoravel(lado: Side, limite: Decimal, preco: Decimal) -> bool:
        """Indica se ``preco`` respeita o ``limite`` de uma ordem agressora.

        Compra aceita preço ``<=`` ao seu limite; venda aceita ``>=``.
        """
        if lado is Side.BUY:
            return preco <= limite
        return preco >= limite

    def _cruzaria(self, order: Order, preco: Decimal) -> bool:
        """Indica se ``order``, posta em ``preco``, executaria contra o livro."""
        oposto = self.asks if order.side is Side.BUY else self.bids
        melhor = oposto.best_price()
        return melhor is not None and self._favoravel(order.side, preco, melhor)

    def _registrar_trade(self, preco: Decimal | None, qtd: int) -> None:
        """Acrescenta uma linha em ``trades``. Ignora execuções vazias."""
        if preco is not None and qtd > 0:
            self.trades.append(f"Trade, price: {preco}, qty: {qtd}")