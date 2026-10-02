"""Núcleo do livro de ofertas (order book) da Matching Engine.

Premissas de design:
    * Preços sempre em ``decimal.Decimal`` (nunca ``float``).
    * Prioridade temporal por ``seq`` inteiro e determinístico, gerado
      externamente pela engine (``itertools.count``).
    * Ordens Pegged nunca são reprecificadas fisicamente: ficam em uma fila
      própria e herdam o preço do melhor nível de ordens reais do mesmo lado.
"""

from __future__ import annotations

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
        qty: Quantidade original informada.
        seq: Número de sequência que define a prioridade temporal.
        price: Preço limite. ``None`` para market e peg.
        leaves_qty: Quantidade restante executável. Inicia igual a ``qty``.
    """

    id: str
    side: Side
    type: OrderType
    qty: int
    seq: int
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
            via ``bisect`` nos commits seguintes). O melhor bid fica em
            ``[-1]`` e o melhor ask em ``[0]``.
        pegs: Fila FIFO das ordens Pegged deste lado, separada dos níveis.

    Invariante (a ser garantida pelos métodos de inserção/remoção):
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

    # ------------------------------------------------------------------
    # Ordens limit reais (níveis de preço)
    # ------------------------------------------------------------------

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
        preço é removido de ``prices`` na mesma operação. Assim, todo preço em
        ``prices`` sempre possui ao menos uma ordem.

        A localização do preço usa busca binária (O(log L)); a remoção do
        vetor em si continua O(L) por deslocamento de memória, como esperado.

        Args:
            order: Ordem LIMIT atualmente presente neste lado do livro.

        Raises:
            KeyError: se o nível ou a ordem não existirem neste lado.
        """
        preco = order.price
        fila = self.levels[preco]  # KeyError se o nível não existir
        del fila[order.id]  # KeyError se a ordem não estiver no nível

        # Se, ao remover a ordem, nn houver mas nenhuma no seu preço, 
        # apaga a fila de tal preço

        if not fila:
            del self.levels[preco]
            del self.prices[bisect_left(self.prices, preco)]

    # ------------------------------------------------------------------
    # Ordens pegged (fila VIP isolada)
    # ------------------------------------------------------------------

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