"""Testes automatizados da Matching Engine (pytest).

Cobrem matching e FIFO, prioridade no amend, escudo de spread, proteção de
liquidez (``leaves_qty``), pegs (execução, prioridade, dormência) e a camada
de I/O. Toda a suíte roda com a verificação automática das invariantes do
livro ao final de cada teste (fixture ``engine``).
"""

from __future__ import annotations

import io
import sys
from decimal import Decimal

import pytest

import matching_engine
from matching_engine import BookSide, MatchingEngine, Order, OrderType, Side

BUY, SELL = Side.BUY, Side.SELL


# ----------------------------------------------------------------------
# Fábricas e utilitários
# ----------------------------------------------------------------------


def limit(oid: str, side: Side, price: str, qty: int) -> Order:
    """Cria uma ordem LIMIT (preço informado como string, convertido em Decimal)."""
    return Order(oid, side, OrderType.LIMIT, qty, price=Decimal(price))


def market(oid: str, side: Side, qty: int) -> Order:
    """Cria uma ordem MARKET."""
    return Order(oid, side, OrderType.MARKET, qty)


def peg(oid: str, side: Side, qty: int) -> Order:
    """Cria uma ordem PEG (sem preço próprio)."""
    return Order(oid, side, OrderType.PEG, qty)


def place(engine: MatchingEngine, *ordens: Order):
    """Envia ordens à engine em sequência; devolve a ordem (ou a tupla delas)."""
    for ordem in ordens:
        engine.process_order(ordem)
    return ordens[0] if len(ordens) == 1 else ordens


def ids_do_nivel(lado: BookSide, preco: str) -> list[str]:
    """Ids das ordens de um nível, na ordem da fila (prioridade)."""
    return list(lado.levels[Decimal(preco)])


def _snapshot(engine: MatchingEngine) -> dict:
    """Foto comparável do estado completo do livro (para provar 'intacto')."""

    def lado(bs: BookSide) -> dict:
        return {
            "prices": list(bs.prices),
            "levels": {
                p: [(o.id, o.seq, o.qty, o.leaves_qty, o.price) for o in fila.values()]
                for p, fila in bs.levels.items()
            },
            "pegs": [(o.id, o.seq, o.qty, o.leaves_qty) for o in bs.pegs.values()],
        }

    return {
        "bids": lado(engine.bids),
        "asks": lado(engine.asks),
        "index": {k: (v.seq, v.qty, v.leaves_qty, v.price) for k, v in engine.index.items()},
        "trades": list(engine.trades),
    }


def _verificar_invariantes(engine: MatchingEngine) -> None:
    """Falha se qualquer invariante estrutural do livro estiver violada."""
    repousando: dict[str, Order] = {}

    for lado in (engine.bids, engine.asks):
        assert lado.prices == sorted(set(lado.prices)), "prices deve ser crescente e único"
        assert set(lado.levels) == set(lado.prices), "levels e prices divergem"
        for preco, fila in lado.levels.items():
            assert fila, f"nível vazio não removido: {preco}"
            seqs = [o.seq for o in fila.values()]
            assert seqs == sorted(seqs), "FIFO violado dentro do nível"
            for o in fila.values():
                assert o.type is OrderType.LIMIT and o.price == preco and o.side is lado.side
                assert 0 < o.leaves_qty <= o.qty
                assert o.id not in repousando, "ordem em dois lugares"
                repousando[o.id] = o
        seqs = [o.seq for o in lado.pegs.values()]
        assert seqs == sorted(seqs), "FIFO violado na fila de pegs"
        for o in lado.pegs.values():
            assert o.is_peg and o.side is lado.side and o.price is None
            assert 0 < o.leaves_qty <= o.qty
            assert o.id not in repousando, "ordem em dois lugares"
            repousando[o.id] = o

    assert set(engine.index) == set(repousando), "index diverge do livro"
    assert all(engine.index[i] is o for i, o in repousando.items())
    assert len({o.seq for o in repousando.values()}) == len(repousando), "seq duplicado"

    melhor_bid, melhor_ask = engine.bids.best_price(), engine.asks.best_price()
    if melhor_bid is not None and melhor_ask is not None:
        assert melhor_bid < melhor_ask, "livro cruzado"


@pytest.fixture
def engine():
    """Engine nova; ao final do teste, confere as invariantes do livro."""
    e = MatchingEngine()
    yield e
    _verificar_invariantes(e)


@pytest.fixture
def livro_com_spread(engine):
    """Livro com bid 10 x10 (id 'b') e ask 11 x10 (id 's')."""
    place(engine, limit("b", BUY, "10", 10), limit("s", SELL, "11", 10))
    return engine


@pytest.fixture
def ask_parcial(engine):
    """Dois asks @20 de 100; um market buy de 60 deixa 'a' com 40 restantes."""
    a, b = place(engine, limit("a", SELL, "20", 100), limit("b", SELL, "20", 100))
    place(engine, market("m", BUY, 60))
    assert (a.qty, a.leaves_qty) == (100, 40)
    return a, b


# ----------------------------------------------------------------------
# 1. Matching básico e FIFO
# ----------------------------------------------------------------------


class TestMatchingBasico:
    def test_exemplo_do_enunciado(self, engine):
        place(
            engine,
            limit("1", BUY, "10", 100),
            limit("2", SELL, "20", 100),
            limit("3", SELL, "20", 200),
            market("4", BUY, 150),
            market("5", BUY, 200),
            market("6", SELL, 200),
        )
        assert engine.trades == [
            "Trade, price: 20, qty: 150",
            "Trade, price: 20, qty: 150",
            "Trade, price: 10, qty: 100",
        ]
        assert engine.index == {}

    def test_fills_no_mesmo_preco_sao_consolidados(self, engine):
        a, b = place(engine, limit("a", SELL, "20", 100), limit("b", SELL, "20", 200))
        place(engine, market("m", BUY, 150))
        assert engine.trades == ["Trade, price: 20, qty: 150"]
        assert "a" not in engine.index
        assert b.leaves_qty == 150

    def test_precos_diferentes_geram_linhas_separadas(self, engine):
        place(
            engine,
            limit("a", SELL, "10", 50),
            limit("b", SELL, "11", 50),
            limit("c", SELL, "12", 50),
        )
        place(engine, market("m", BUY, 120))
        assert engine.trades == [
            "Trade, price: 10, qty: 50",
            "Trade, price: 11, qty: 50",
            "Trade, price: 12, qty: 20",
        ]
        assert engine.index["c"].leaves_qty == 30

    def test_consolida_por_preco_em_execucao_mista(self, engine):
        place(
            engine,
            limit("a", SELL, "10", 30),
            limit("b", SELL, "10", 30),
            limit("c", SELL, "11", 40),
        )
        place(engine, market("m", BUY, 80))
        assert engine.trades == ["Trade, price: 10, qty: 60", "Trade, price: 11, qty: 20"]
        assert engine.index["c"].leaves_qty == 20

    def test_fifo_dentro_do_nivel(self, engine):
        a, b, c = place(
            engine,
            limit("a", SELL, "20", 30),
            limit("b", SELL, "20", 30),
            limit("c", SELL, "20", 30),
        )
        place(engine, market("m", BUY, 45))
        assert "a" not in engine.index
        assert b.leaves_qty == 15
        assert c.leaves_qty == 30
        assert ids_do_nivel(engine.asks, "20") == ["b", "c"]

    def test_prioridade_de_preco_vence_a_de_tempo(self, engine):
        place(engine, limit("velha", SELL, "21", 10), limit("nova", SELL, "20", 10))
        place(engine, market("m", BUY, 5))
        assert engine.trades == ["Trade, price: 20, qty: 5"]
        assert engine.index["velha"].leaves_qty == 10

    def test_saldo_de_market_evapora(self, engine):
        place(engine, limit("a", SELL, "20", 150), market("m", BUY, 200))
        assert engine.trades == ["Trade, price: 20, qty: 150"]
        assert engine.index == {}
        assert engine.bids.prices == [] and engine.asks.prices == []

    def test_market_em_livro_vazio_nao_gera_trade_nem_erro(self, engine):
        place(engine, market("m", SELL, 10))
        assert engine.trades == []
        assert engine.index == {}

    def test_limit_agressiva_executa_ao_preco_passivo_e_repousa_o_saldo(self, engine):
        place(engine, limit("a", SELL, "10", 50), limit("b", BUY, "11", 80))
        assert engine.trades == ["Trade, price: 10, qty: 50"]
        assert engine.index["b"].leaves_qty == 30
        assert engine.bids.best_price() == Decimal("11")

    def test_limit_agressiva_para_no_preco_limite(self, engine):
        place(engine, limit("a", SELL, "10", 50), limit("c", SELL, "12", 50))
        place(engine, limit("b", BUY, "11", 80))
        assert engine.trades == ["Trade, price: 10, qty: 50"]
        assert engine.index["c"].leaves_qty == 50
        assert engine.bids.best_price() == Decimal("11")

    def test_limit_que_nao_cruza_apenas_repousa(self, engine):
        place(engine, limit("a", SELL, "10", 5), limit("b", BUY, "9", 5))
        assert engine.trades == []
        assert set(engine.index) == {"a", "b"}

    def test_id_duplicado_e_rejeitado(self, engine):
        place(engine, limit("x", BUY, "10", 1))
        with pytest.raises(ValueError, match="duplicado"):
            place(engine, limit("x", BUY, "9", 1))


# ----------------------------------------------------------------------
# 2. Prioridade no amend
# ----------------------------------------------------------------------


class TestAmendPrioridade:
    @pytest.fixture
    def dois_asks(self, engine):
        return place(engine, limit("a", SELL, "20", 100), limit("b", SELL, "20", 100))

    def test_reducao_mantem_seq_e_posicao(self, engine, dois_asks):
        a, b = dois_asks
        seq_original = a.seq

        engine.amend_order("a", 50)

        assert a.seq == seq_original
        assert ids_do_nivel(engine.asks, "20") == ["a", "b"]
        assert engine.asks.head() == (a, Decimal("20"))
        assert (a.qty, a.leaves_qty) == (50, 50)

        place(engine, market("m", BUY, 60))  # 'a' continua sendo executada primeiro
        assert engine.trades == ["Trade, price: 20, qty: 60"]
        assert "a" not in engine.index
        assert b.leaves_qty == 90

    def test_aumento_de_quantidade_vai_para_o_fim_da_fila(self, engine, dois_asks):
        a, b = dois_asks

        engine.amend_order("a", 150)

        assert a.seq > b.seq
        assert ids_do_nivel(engine.asks, "20") == ["b", "a"]
        assert engine.asks.head()[0] is b
        assert a.leaves_qty == 150

        place(engine, market("m", BUY, 100))  # agora 'b' é executada primeiro
        assert "b" not in engine.index
        assert a.leaves_qty == 150

    def test_mudanca_de_preco_vai_para_o_fim_do_novo_nivel(self, engine):
        a, b, c = place(
            engine,
            limit("a", SELL, "20", 10),
            limit("b", SELL, "20", 10),
            limit("c", SELL, "21", 10),
        )

        engine.amend_order("a", 10, Decimal("21"))

        assert ids_do_nivel(engine.asks, "21") == ["c", "a"]
        assert ids_do_nivel(engine.asks, "20") == ["b"]
        assert a.price == Decimal("21") and a.seq > c.seq

    def test_mudanca_de_preco_reposiciona_na_faixa_e_remove_nivel_vazio(self, engine):
        """Exemplo do enunciado: 200 @ 10 passa para 9.98 e perde prioridade."""
        b1, b2, _ = place(
            engine,
            limit("b1", BUY, "10", 200),
            limit("b2", BUY, "9.99", 100),
            limit("a1", SELL, "10.5", 100),
        )

        engine.amend_order("b1", 200, Decimal("9.98"))

        assert engine.bids.prices == [Decimal("9.98"), Decimal("9.99")]
        assert Decimal("10") not in engine.bids.levels  # nível vazio removido
        assert engine.bids.best_price() == Decimal("9.99")
        assert ids_do_nivel(engine.bids, "9.98") == ["b1"]
        assert b1.price == Decimal("9.98")
        assert engine.trades == []

    @pytest.mark.parametrize(
        "new_qty, new_price",
        [(150, None), (100, "19"), (150, "19")],
        ids=["so-quantidade", "so-preco", "quantidade-e-preco"],
    )
    def test_perda_de_prioridade_renova_seq_deterministicamente(
        self, engine, dois_asks, new_qty, new_price
    ):
        a, b = dois_asks
        preco = Decimal(new_price) if new_price else None

        engine.amend_order("a", new_qty, preco)

        assert a.seq == b.seq + 1  # exatamente um seq novo consumido
        assert a.leaves_qty == new_qty

    def test_preco_explicito_igual_ao_atual_nao_conta_como_mudanca(self, engine, dois_asks):
        a, _ = dois_asks
        seq_original = a.seq

        engine.amend_order("a", 60, Decimal("20"))

        assert a.seq == seq_original
        assert ids_do_nivel(engine.asks, "20") == ["a", "b"]

    def test_amend_sem_mudanca_e_noop_e_nao_consome_seq(self, engine, dois_asks):
        a, b = dois_asks
        antes = _snapshot(engine)

        engine.amend_order("a", 100)
        engine.amend_order("a", 100, Decimal("20"))

        assert _snapshot(engine) == antes
        novo = place(engine, limit("c", SELL, "20", 1))
        assert novo.seq == b.seq + 1  # nenhum seq foi gasto pelos no-ops

    def test_amend_de_ordem_inexistente_levanta_keyerror(self, engine):
        with pytest.raises(KeyError):
            engine.amend_order("nope", 10)

    def test_amend_de_market_nao_e_possivel(self, engine):
        place(engine, market("m", BUY, 10))  # market nunca repousa
        with pytest.raises(KeyError):
            engine.amend_order("m", 5)


# ----------------------------------------------------------------------
# 3. Escudo de spread
# ----------------------------------------------------------------------


class TestEscudoDeSpread:
    @pytest.mark.parametrize(
        "new_qty, novo_preco",
        [(10, "11"), (10, "12"), (25, "11")],
        ids=["igual-ao-ask", "acima-do-ask", "com-aumento-de-qtd"],
    )
    def test_compra_que_cruzaria_e_rejeitada_e_livro_fica_intacto(
        self, livro_com_spread, new_qty, novo_preco
    ):
        engine = livro_com_spread
        antes = _snapshot(engine)

        with pytest.raises(ValueError, match="cruzaria"):
            engine.amend_order("b", new_qty, Decimal(novo_preco))

        assert _snapshot(engine) == antes
        assert engine.trades == []

    @pytest.mark.parametrize("novo_preco", ["10", "9"], ids=["igual-ao-bid", "abaixo-do-bid"])
    def test_venda_que_cruzaria_e_rejeitada_e_livro_fica_intacto(
        self, livro_com_spread, novo_preco
    ):
        engine = livro_com_spread
        antes = _snapshot(engine)

        with pytest.raises(ValueError, match="cruzaria"):
            engine.amend_order("s", 10, Decimal(novo_preco))

        assert _snapshot(engine) == antes

    def test_rejeicao_nao_consome_seq(self, livro_com_spread):
        engine = livro_com_spread
        s = engine.index["s"]

        with pytest.raises(ValueError):
            engine.amend_order("b", 10, Decimal("11"))

        novo = place(engine, limit("x", BUY, "9", 1))
        assert novo.seq == s.seq + 1

    def test_preco_logo_abaixo_do_ask_e_aceito_sem_gerar_trade(self, livro_com_spread):
        engine = livro_com_spread

        engine.amend_order("b", 10, Decimal("10.99"))

        assert engine.bids.best_price() == Decimal("10.99")
        assert engine.trades == []

    def test_sem_liquidez_no_lado_oposto_qualquer_preco_e_aceito(self, engine):
        place(engine, limit("b", BUY, "10", 10))

        engine.amend_order("b", 10, Decimal("1000"))

        assert engine.bids.best_price() == Decimal("1000")
        assert engine.trades == []


# ----------------------------------------------------------------------
# 4. Proteção de liquidez (leaves_qty)
# ----------------------------------------------------------------------


class TestProtecaoDeLiquidez:
    def test_aumento_acima_do_saldo_perde_prioridade_mesmo_abaixo_da_qty_original(
        self, engine, ask_parcial
    ):
        a, b = ask_parcial

        # 70 < 100 (qty original), mas 70 > 40 (saldo): é aumento, não redução.
        engine.amend_order("a", 70)

        assert engine.asks.head()[0] is b
        assert ids_do_nivel(engine.asks, "20") == ["b", "a"]
        assert a.leaves_qty == 70
        assert a.qty == 130  # 60 executadas + 70 restantes

    def test_liquidez_ja_executada_nao_ressuscita(self, engine, ask_parcial):
        engine.amend_order("a", 70)

        place(engine, market("m2", BUY, 1000))

        # b (100) + a (70). Nunca 100 + 100, que ressuscitaria os 60 executados.
        assert engine.trades[-1] == "Trade, price: 20, qty: 170"
        assert engine.asks.prices == []

    def test_reducao_abaixo_do_saldo_mantem_prioridade(self, engine, ask_parcial):
        a, _ = ask_parcial
        seq_original = a.seq

        engine.amend_order("a", 30)

        assert a.seq == seq_original
        assert engine.asks.head()[0] is a
        assert (a.qty, a.leaves_qty) == (90, 30)

    def test_new_qty_igual_ao_saldo_e_noop(self, engine, ask_parcial):
        a, _ = ask_parcial
        antes = _snapshot(engine)

        engine.amend_order("a", 40)

        assert _snapshot(engine) == antes

    @pytest.mark.parametrize("new_qty", [30, 40, 70, 500])
    def test_quantidade_executada_permanece_constante(self, engine, ask_parcial, new_qty):
        a, _ = ask_parcial

        engine.amend_order("a", new_qty)

        assert a.leaves_qty == new_qty
        assert a.qty - a.leaves_qty == 60  # executado nunca muda por amend

    @pytest.mark.parametrize("new_qty", [0, -5])
    def test_quantidade_nao_positiva_e_rejeitada_e_livro_fica_intacto(
        self, engine, ask_parcial, new_qty
    ):
        antes = _snapshot(engine)

        with pytest.raises(ValueError, match="cancel_order"):
            engine.amend_order("a", new_qty)

        assert _snapshot(engine) == antes


# ----------------------------------------------------------------------
# 5. Pegs
# ----------------------------------------------------------------------


class TestPegs:
    def test_peg_executa_ao_preco_corrente_do_melhor_nivel_do_proprio_lado(self, engine):
        b1 = place(engine, limit("b1", BUY, "10", 200))
        p = place(engine, peg("p", BUY, 150))
        b2 = place(engine, limit("b2", BUY, "9.99", 100))

        place(engine, market("m", SELL, 250))

        # b1 esgota o nível 10; a referência cai para 9.99 e o peg (seq mais
        # antigo que b2) executa ali, e não ao preço em que foi criado.
        assert engine.trades == ["Trade, price: 10, qty: 200", "Trade, price: 9.99, qty: 50"]
        assert "b1" not in engine.index
        assert p.leaves_qty == 100
        assert b2.leaves_qty == 100

    def test_peg_segue_o_novo_topo_sem_ser_reprecificado(self, engine):
        place(engine, limit("b1", BUY, "10", 200))
        p = place(engine, peg("p", BUY, 150))
        b3 = place(engine, limit("b3", BUY, "10.1", 300))

        assert engine.bids.head() == (p, Decimal("10.1"))  # peg à frente de b3
        assert p.price is None  # nunca reprecificado fisicamente

        place(engine, market("m", SELL, 150))

        assert engine.trades == ["Trade, price: 10.1, qty: 150"]
        assert "p" not in engine.index
        assert b3.leaves_qty == 300

    def test_prioridade_do_peg_respeita_o_seq_dentro_do_nivel(self, engine):
        r1, p, r2 = place(
            engine,
            limit("r1", BUY, "10", 5),
            peg("p", BUY, 5),
            limit("r2", BUY, "10", 5),
        )

        place(engine, market("m", SELL, 7))

        # r1 (seq 1) -> peg (seq 2) -> r2 (seq 3)
        assert engine.trades == ["Trade, price: 10, qty: 7"]
        assert "r1" not in engine.index
        assert p.leaves_qty == 3
        assert r2.leaves_qty == 5

    def test_peg_de_venda_segue_o_melhor_ask(self, engine):
        place(engine, limit("s1", SELL, "10.5", 100))
        p = place(engine, peg("p", SELL, 50))
        s2 = place(engine, limit("s2", SELL, "11", 100))

        place(engine, market("m", BUY, 130))

        # s1 esgota o nível 10.5; o melhor ask passa a 11 e o peg (seq mais
        # antigo que s2) executa antes dela, ao preço 11.
        assert engine.trades == ["Trade, price: 10.5, qty: 100", "Trade, price: 11, qty: 30"]
        assert p.leaves_qty == 20
        assert s2.leaves_qty == 100

    def test_peg_sem_referencia_fica_dormente_sem_erro(self, engine):
        p = place(engine, peg("p", BUY, 10))

        assert engine.bids.head() is None  # dormente
        place(engine, market("m", SELL, 10))  # não há contraparte: sem trade, sem erro
        place(engine, limit("s", SELL, "1", 5))  # nem um ask baratíssimo o executa

        assert engine.trades == []
        assert p.leaves_qty == 10
        assert "p" in engine.index

    def test_peg_nunca_agride_ao_ser_inserido(self, engine):
        place(engine, limit("s", SELL, "10", 5))

        place(engine, peg("p", BUY, 5))

        assert engine.trades == []
        assert set(engine.index) == {"s", "p"}

    def test_lado_oposto_seco_peg_repousa_silenciosamente(self, engine):
        r, p = place(engine, limit("r", BUY, "10", 5), peg("p", BUY, 5))

        assert engine.asks.head() is None
        assert engine.bids.head() == (r, Decimal("10"))
        assert engine.trades == []
        assert set(engine.index) == {"r", "p"}

    def test_peg_dorme_quando_a_referencia_esgota_e_acorda_com_nova_referencia(self, engine):
        place(engine, limit("r", BUY, "10", 10))
        p = place(engine, peg("p", BUY, 5))

        # Esgota a única ordem real: o peg perde a referência e não executa.
        place(engine, limit("s", SELL, "10", 12))
        assert engine.trades == ["Trade, price: 10, qty: 10"]
        assert engine.bids.head() is None
        assert p.leaves_qty == 5
        assert engine.index["s"].leaves_qty == 2  # saldo da agressora repousa

        # Nova referência (seq mais novo que o peg): o peg volta, à frente dela.
        place(engine, limit("r2", BUY, "9", 3))
        assert engine.bids.head() == (p, Decimal("9"))

        place(engine, market("m", SELL, 5))
        assert engine.trades[-1] == "Trade, price: 9, qty: 5"
        assert "p" not in engine.index
        assert engine.index["r2"].leaves_qty == 3

    def test_amend_de_peg_reducao_mantem_posicao_e_aumento_vai_ao_fim(self, engine):
        place(engine, limit("r", BUY, "10", 5))
        p1, p2 = place(engine, peg("p1", BUY, 10), peg("p2", BUY, 10))

        engine.amend_order("p1", 4)
        assert list(engine.bids.pegs) == ["p1", "p2"]

        engine.amend_order("p1", 20)
        assert list(engine.bids.pegs) == ["p2", "p1"]
        assert p1.seq > p2.seq

    def test_amend_de_peg_com_preco_e_rejeitado_e_livro_fica_intacto(self, engine):
        place(engine, limit("r", BUY, "10", 5), peg("p", BUY, 10))
        antes = _snapshot(engine)

        with pytest.raises(ValueError, match="PEG"):
            engine.amend_order("p", 5, Decimal("10"))

        assert _snapshot(engine) == antes

    def test_cancelamento_de_peg(self, engine):
        place(engine, limit("r", BUY, "10", 5), peg("p", BUY, 10))

        engine.cancel_order("p")

        assert engine.bids.pegs == {}
        assert "p" not in engine.index


# ----------------------------------------------------------------------
# 6. Cancelamento, I/O e tratamento de erro
# ----------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch, capsys):
    """Executa ``main()`` com as linhas dadas em ``stdin``; devolve o stdout."""

    def executar(*linhas: str) -> list[str]:
        monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(linhas) + "\n"))
        matching_engine.main()
        return capsys.readouterr().out.splitlines()

    return executar


class TestCancelamentoEIO:
    def test_cancelar_ordem_inexistente_levanta_keyerror(self, engine):
        with pytest.raises(KeyError):
            engine.cancel_order("nope")

    def test_cancelar_duas_vezes_levanta_keyerror(self, engine):
        place(engine, limit("a", BUY, "10", 5))
        engine.cancel_order("a")
        with pytest.raises(KeyError):
            engine.cancel_order("a")

    def test_cancelar_ordem_totalmente_executada_levanta_keyerror(self, engine):
        place(engine, limit("a", SELL, "10", 5), market("m", BUY, 5))
        with pytest.raises(KeyError):
            engine.cancel_order("a")

    def test_cancelar_remove_nivel_vazio_e_atualiza_o_melhor_preco(self, engine):
        place(engine, limit("a", BUY, "10", 5), limit("b", BUY, "9", 5))

        engine.cancel_order("a")

        assert engine.bids.prices == [Decimal("9")]
        assert engine.bids.best_price() == Decimal("9")
        assert "a" not in engine.index

    def test_cli_erro_em_cancel_nao_derruba_a_sessao(self, cli):
        saida = cli("cancel order nope", "limit buy 10 5 x", "cancel order x")
        assert saida == [
            "Erro: ordem inexistente ou já finalizada: nope",
            "Order created: buy 5 @ 10 x",
            "Order cancelled",
        ]

    def test_cli_tolera_prompt_e_normaliza_preco(self, cli):
        saida = cli(">>> limit buy 10.0 100", ">>> limit sell 10 40")
        assert saida == [
            "Order created: buy 100 @ 10 identificador_1",
            "Order created: sell 40 @ 10 identificador_2",
            "Trade, price: 10, qty: 40",
        ]

    def test_cli_entradas_invalidas_sao_reportadas_sem_interromper(self, cli):
        saida = cli("foo", "limit buy abc 1", "limit buy 10 0", "limit buy 10 5 ok")
        assert saida[0].startswith("Erro: comando desconhecido")
        assert saida[1].startswith("Erro: preço inválido")
        assert saida[2].startswith("Erro: quantidade deve ser positiva")
        assert saida[3] == "Order created: buy 5 @ 10 ok"

    def test_cli_print_do_livro_com_peg_na_posicao_de_prioridade(self, cli):
        saida = cli(
            "limit buy 10 200 b1",
            "limit buy 9.99 100 b2",
            "limit sell 10.5 100 a1",
            "peg bid buy 150 p1",
            "limit buy 10.1 300 b3",
            "print",
        )
        assert saida[-8:] == [
            "Ordens de Venda (Asks)",
            "  100 @ 10.5  a1",
            "-" * 40,
            "Ordens de Compra (Bids)",
            "  150 @ 10.1  p1  [peg]",
            "  300 @ 10.1  b3",
            "  200 @ 10  b1",
            "  100 @ 9.99  b2",
        ]