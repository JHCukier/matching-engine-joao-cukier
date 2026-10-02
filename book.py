"""Núcleo do livro de ofertas (order book) da Matching Engine.

Commit 1: modelo de ordem (`Order`) e estrutura de um lado do livro
(`BookSide`), com rastreio do melhor preço e da ordem na cabeça da fila.

Premissas de design:
    * Preços sempre em ``decimal.Decimal`` (nunca ``float``).
    * Prioridade temporal por ``seq`` inteiro e determinístico, gerado
      externamente pela engine (``itertools.count``).
    * Ordens Pegged nunca são reprecificadas fisicamente: ficam em uma fila
      própria e herdam o preço do melhor nível de ordens reais do mesmo lado.
"""

from __future__ import annotations

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

"""Em vez de escrever o clássico def __init__(self, id, side...): e gastar 10 linhas, 
   o decorador @dataclass gera o construtor automaticamente.
   
   Cada objeto criado carrega um dicionário oculto na memória para permitir a adição de
   novas variáveis em tempo de execução. Isso consome muita RAM. O slots=True bloqueia isso, 
   travando a estrutura da classe e economizando muita memória
"""

@dataclass(slots=True, eq=False)
class Order:
    """Ordem do livro.

    ``eq=False`` mantém a comparação por identidade (a ordem é uma entidade,
    não um valor) e preserva o ``__hash__`` padrão.
    Uma ordem só é igual à outra se for o exato mesmo objeto no mesmo endereço de memória.
    Isso é vital para usarmos as ordens como chaves seguras no nosso dicionário global.

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

    """
    Trava a variável (com init=False) para impedir que qualquer se
    crie uma ordem já dizendo que o saldo restante é diferente da quantidade original.
    """
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
