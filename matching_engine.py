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
from enum import Enum
import sys
from decimal import Decimal, DecimalException
from decimal import Decimal, InvalidOperation


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
#elif comando == "cancel":
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
            ValueError: se o ``id`` já estiver em uso por uma ordem no livro,
                se ``qty`` não for inteiro positivo, se o preço de uma LIMIT
                não for ``Decimal`` finito e positivo ou se MARKET/PEG
                trouxerem preço. A validação ocorre antes de consumir ``seq``
                e de qualquer mutação do livro.
        """
        if incoming.id in self.index:
            raise ValueError(f"id de ordem duplicado: {incoming.id}")

        # Validação de fronteira: antes de consumir seq ou tocar no livro.
        if not isinstance(incoming.qty, int) or isinstance(incoming.qty, bool) or incoming.qty <= 0:
            raise ValueError("qty deve ser um inteiro positivo")
        if incoming.type is OrderType.LIMIT:
            if not isinstance(incoming.price, Decimal) or not incoming.price.is_finite() or incoming.price <= 0:
                raise ValueError("preço de LIMIT deve ser um Decimal finito e positivo")
        elif incoming.price is not None:
            raise ValueError("ordens MARKET e PEG não possuem preço")

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
        spread é rejeitado. Os argumentos são validados na entrada (tipo,
        finitude e positividade) e toda validação de negócio ocorre antes de
        qualquer mutação, então uma chamada rejeitada deixa o livro intacto.
        Isso não equivale a rollback geral: uma exceção inesperada *depois* da
        validação (violação interna) não é desfeita.

        Args:
            order_id: Identificador da ordem em repouso.
            new_qty: Nova quantidade restante (``int`` > 0).
            new_price: Novo preço (``Decimal`` finito e > 0). ``None`` mantém o
                preço atual. Não se aplica a ordens PEG.

        Raises:
            KeyError: se a ordem não existir em repouso (market nunca repousa).
            ValueError: se ``new_qty`` não for inteiro positivo, se
                ``new_price`` não for ``Decimal`` finito e positivo, se
                informar preço para um PEG ou se o novo preço cruzaria o spread.
        """
        # Validação de fronteira: antes de qualquer lookup ou mutação.
        if not isinstance(new_qty, int) or isinstance(new_qty, bool) or new_qty <= 0:
            raise ValueError("new_qty deve ser um inteiro positivo; use cancel_order")
        if new_price is not None:
            if not isinstance(new_price, Decimal):
                raise ValueError("new_price deve ser um Decimal")
            if not new_price.is_finite() or new_price <= 0:  # exclui NaN e infinitos
                raise ValueError("new_price deve ser finito e positivo")

        order = self.index[order_id]  # O(1); KeyError se não existir
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

        executado = order.qty - order.leaves_qty #o que já foi executado

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
            self.trades.append(f"Trade, price: {preco:f}, qty: {qtd}")

# ----------------------------------------------------------------------
# Interface de terminal (I/O)
# ----------------------------------------------------------------------


def _ordens_do_lado(lado: BookSide) -> list[tuple[Decimal | None, Order]]:
    """Lista as ordens de um lado na ordem de exibição, do maior ao menor preço.

    Dentro de cada nível a ordem é a de prioridade (FIFO por ``seq``). Os pegs
    herdam o melhor preço do lado e são intercalados com as ordens reais desse
    nível pelo ``seq``. Pegs sem referência (dormentes) vêm ao final, sem preço.

    Não altera o estado do livro.
    """
    melhor = lado.best_price()
    resultado: list[tuple[Decimal | None, Order]] = []

    for preco in reversed(lado.prices):  # prices é crescente
        ordens = list(lado.levels[preco].values())  
        if preco == melhor:
            ordens = sorted(ordens + list(lado.pegs.values()), key=lambda o: o.seq)
        resultado.extend((preco, o) for o in ordens)

    if melhor is None:
        resultado.extend((None, p) for p in lado.pegs.values())
    return resultado


def _formatar_ordem(preco: Decimal | None, ordem: Order) -> str:
    """Formata uma linha do livro: ``<restante> @ <preço>  <id>``."""
    if ordem.is_peg:
        marca = "  [peg]" if preco is not None else "  [peg dormente]"
    else:
        marca = ""
    preco_txt = "--" if preco is None else f"{preco:f}"
    return f"  {ordem.leaves_qty} @ {preco_txt}  {ordem.id}{marca}"


def print_book(engine: MatchingEngine) -> None:
    """Imprime o livro: vendas (maior -> menor preço), depois compras (idem).

    Cada linha mostra a quantidade restante (``leaves_qty``), o preço e o id.
    """
    print("Ordens de Venda (Asks)")
    linhas = _ordens_do_lado(engine.asks)
    for preco, ordem in linhas:
        print(_formatar_ordem(preco, ordem))
    if not linhas:
        print("  (vazio)")

    print("-" * 40)

    print("Ordens de Compra (Bids)")
    linhas = _ordens_do_lado(engine.bids)
    for preco, ordem in linhas:
        print(_formatar_ordem(preco, ordem))
    if not linhas:
        print("  (vazio)")


def _parse_lado(token: str) -> Side:
    """Converte ``buy``/``sell`` (sem distinguir maiúsculas) em ``Side``."""
    try:
        return Side(token.lower())
    except ValueError:
        raise ValueError(f"lado inválido: {token!r} (use buy ou sell)") from None


def _parse_qtd(token: str) -> int:
    """Converte o token em quantidade inteira positiva."""
    try:
        qtd = int(token)
    except ValueError:
        raise ValueError(f"quantidade inválida: {token!r}") from None
    if qtd <= 0:
        raise ValueError("quantidade deve ser positiva")
    return qtd


# Domínio de preços aceito pela CLI (ver ``_parse_preco``).
_PRECO_TOKEN_MAX = 40  # caracteres do token
_PRECO_INTEIROS_MAX = 12  # dígitos da parte inteira
_PRECO_DECIMAIS_MAX = 8  # casas decimais reais (zeros à direita não contam)


def _parse_preco(token: str) -> Decimal:
    """Converte o token em ``Decimal`` exato, positivo e de domínio limitado.

    O valor nunca é arredondado: o ``Decimal`` devolvido é exatamente o
    informado, a menos de zeros à direita, removidos manipulando o coeficiente
    (aritmética inteira, sem passar pelo contexto decimal). Entradas fora do
    domínio são rejeitadas, nunca truncadas.

    Domínio aceito: finito, estritamente positivo, até 12 dígitos inteiros e
    até 8 casas decimais reais. Isso limita o preço a 20 dígitos significativos
    (abaixo dos 28 do contexto padrão) e impede underflow para zero e
    expansão de memória por expoentes absurdos (``1e999999``).

    Raises:
        ValueError: token inválido ou fora do domínio aceito.
    """
    if len(token) > _PRECO_TOKEN_MAX:
        raise ValueError(f"preço longo demais (máx. {_PRECO_TOKEN_MAX} caracteres)")
    try:
        preco = Decimal(token)
    except InvalidOperation:
        raise ValueError(f"preço inválido: {token!r}") from None
    if not preco.is_finite() or preco <= 0:
        raise ValueError(f"preço deve ser finito e positivo: {token!r}")

    _, digitos, expoente = preco.as_tuple()
    digitos = list(digitos)
    while len(digitos) > 1 and digitos[-1] == 0:  # zeros à direita não contam
        digitos.pop()
        expoente += 1

    if -expoente > _PRECO_DECIMAIS_MAX:
        raise ValueError(f"preço aceita no máximo {_PRECO_DECIMAIS_MAX} casas decimais: {token!r}")
    if len(digitos) + expoente > _PRECO_INTEIROS_MAX:
        raise ValueError(f"preço aceita no máximo {_PRECO_INTEIROS_MAX} dígitos inteiros: {token!r}")

    if expoente > 0:  # forma canônica sem notação científica: 1E+2 -> 100
        digitos.extend([0] * expoente)
        expoente = 0
    return Decimal((0, tuple(digitos), expoente))


def _gerar_id(engine: MatchingEngine, contador: "itertools.count[int]") -> str:
    """Gera um id sequencial que não colida com ordens em repouso."""
    while True:
        candidato = f"identificador_{next(contador)}"
        if candidato not in engine.index:
            return candidato


def _submeter(engine: MatchingEngine, ordem: Order, resumo: str | None) -> None:
    """Envia a ordem à engine e imprime a confirmação e os trades gerados.

    Args:
        engine: Motor de matching.
        ordem: Ordem já validada.
        resumo: Texto de ``Order created`` (sem o id). ``None`` suprime a
            confirmação (usado por market, que nunca repousa).
    """
    antes = len(engine.trades)
    engine.process_order(ordem)
    if resumo is not None:
        print(f"Order created: {resumo} {ordem.id}")
    for trade in engine.trades[antes:]:
        print(trade)


def _executar_comando(
    engine: MatchingEngine, contador: "itertools.count[int]", linha: str
) -> None:
    """Interpreta e executa uma linha de comando.

    Comandos:
        limit <side> <price> <qty> [order_id]
        market <side> <qty> [order_id]
        peg [bid|offer] <side> <qty> [order_id]
        cancel [order] <order_id>
        amend <order_id> <new_qty> [new_price]
        print

    Raises:
        ValueError: comando desconhecido, argumentos inválidos ou regra de
            negócio violada (ordem inexistente, cruzamento no amend etc.).
    """
    tokens = linha.split()
    cmd, args = tokens[0].lower(), tokens[1:]

    if cmd == "limit":
        if len(args) not in (3, 4):
            raise ValueError("uso: limit <buy|sell> <price> <qty> [order_id]")
        lado = _parse_lado(args[0])
        preco = _parse_preco(args[1])
        qtd = _parse_qtd(args[2])
        oid = args[3] if len(args) == 4 else _gerar_id(engine, contador)
        ordem = Order(oid, lado, OrderType.LIMIT, qtd, price=preco)
        _submeter(engine, ordem, f"{lado.value} {qtd} @ {preco:f}")

    elif cmd == "market":
        if len(args) not in (2, 3):
            raise ValueError("uso: market <buy|sell> <qty> [order_id]")
        lado = _parse_lado(args[0])
        qtd = _parse_qtd(args[1])
        oid = args[2] if len(args) == 3 else _gerar_id(engine, contador)
        _submeter(engine, Order(oid, lado, OrderType.MARKET, qtd), None)
#_formatar_ordem
    elif cmd == "peg":
        ref = None
        if args and args[0].lower() in ("bid", "offer"):
            ref, args = args[0].lower(), args[1:]
        if len(args) not in (2, 3):
            raise ValueError("uso: peg [bid|offer] <buy|sell> <qty> [order_id]")
        lado = _parse_lado(args[0])
        qtd = _parse_qtd(args[1])
        # O peg segue o próprio lado: bid <-> buy, offer <-> sell.
        esperado = "bid" if lado is Side.BUY else "offer"
        if ref is not None and ref != esperado:
            raise ValueError(
                "peg bid exige lado buy e peg offer exige lado sell "
                "(o contrário cruzaria o spread)"
            )
        oid = args[2] if len(args) == 3 else _gerar_id(engine, contador)
        _submeter(
            engine, Order(oid, lado, OrderType.PEG, qtd), f"{lado.value} {qtd} @ peg {esperado}"
        )

    elif cmd == "cancel":
        if len(args) == 2 and args[0].lower() == "order":
            args = args[1:]
        if len(args) != 1:
            raise ValueError("uso: cancel [order] <order_id>")
        try:
            engine.cancel_order(args[0])
            print("Order cancelled")
        except KeyError:
            raise ValueError(f"ordem inexistente ou já finalizada: {args[0]}")

    elif cmd == "amend":
        if len(args) not in (2, 3):
            raise ValueError("uso: amend <order_id> <new_qty> [new_price]")
        qtd = _parse_qtd(args[1])
        preco = _parse_preco(args[2]) if len(args) == 3 else None
        try:
            engine.amend_order(args[0], qtd, preco)
        except KeyError:
            raise ValueError(f"ordem inexistente ou já finalizada: {args[0]}") from None
        print("Order amended")

    elif cmd == "print":
        print_book(engine)

    else:
        raise ValueError(f"comando desconhecido: {tokens[0]!r}")


def main() -> None:
    """Lê comandos de ``sys.stdin`` linha a linha e os executa na engine.

    Erros de um comando são impressos e não interrompem a sessão. O programa
    encerra ao fim da entrada (EOF) ou com Ctrl+C.
    """
    engine = MatchingEngine()
    contador = itertools.count(1)

    try:
        for linha in sys.stdin:
            # Tolera o prompt ">>>" do enunciado ao colar exemplos.
            linha = linha.strip().lstrip(">").strip()
            if not linha:
                continue
            try:
                _executar_comando(engine, contador, linha)
            except (ValueError, DecimalException) as e:
                # Erro de entrada/negócio: reporta e segue a sessão.
                print(f"Erro: {e}")
            # Qualquer outra exceção (KeyError e RuntimeError inclusive: o
            # KeyError esperado já virou ValueError em _executar_comando)
            # propaga de propósito: fail-fast, o processo cai.
    except KeyboardInterrupt:
        print()
    finally:
        sys.stdout.flush()


if __name__ == "__main__":
    main()