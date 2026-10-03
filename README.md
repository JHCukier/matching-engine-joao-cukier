# Python Matching Engine: Price-Time Priority (FIFO)

Este repositório contém a implementação de um Motor de Correspondência de Ordens (Matching Engine) para um único ativo, desenvolvido em Python. O sistema suporta ordens Limit, Market e Pegged, garantindo prioridade estrita de Preço-Tempo (FIFO) e proteção contra corrupção de estado (crossed books e ressurreição de liquidez).

O foco deste design é a resiliência, precisão financeira e latência algorítmica. Abaixo, detalho as principais decisões arquiteturais que orientaram a construção do motor.

## 1. O Panorama Geral (Anatomia do Livro de Ofertas)

A arquitetura do motor foi projetada para lidar com uma contradição inerente às exigências do desafio: como manter a prioridade estrita de fila (FIFO), permitir cancelamentos instantâneos de IDs arbitrários e imprimir o livro perfeitamente ordenado sem sobrecarregar a CPU a cada operação?

A solução foi separar as responsabilidades dentro da classe `BookSide` (o hemisfério de Compra ou Venda), dividindo a estrutura de dados em componentes matemáticos dedicados:

* **A Cisão entre `prices` (Vetor) e `levels` (Mapa):**
  * **O Problema:** Se usássemos apenas um dicionário, perderíamos a capacidade de saber rapidamente qual é o melhor preço (dicionários não ordenam valores). Se usássemos apenas uma lista simples, cancelar uma ordem exigiria varrer tudo em `O(N)`.
  * **A Solução:** Separamos a Prioridade de Preço da Prioridade de Tempo. O vetor `prices` gerencia os preços estritamente via busca binária (`bisect`), o que garante a localização de qualquer nível de preço em `O(log N)`. Como a lista está sempre ordenada, consultar o topo do livro custa absolutos `O(1)` (basta ler a ponta do vetor). Paralelamente, o dicionário `levels` mapeia o preço para uma fila `OrderedDict`, permitindo remover qualquer ordem no meio da fila também em latência `O(1)`.

* **O Isolamento de `pegs` (A Fila Passiva):**
  * **O Problema:** O enunciado exige que ordens Pegged acompanhem o melhor preço do seu lado. Se injetássemos os Pegs dentro das gavetas de `levels`, um sobressalto no mercado obrigaria o motor a varrer o livro e recriar os Pegs em novas gavetas físicas o tempo todo.
  * **A Solução:** Os Pegs foram fisicamente isolados do livro de preços reais. Eles ficam numa "fila de espera VIP" (`pegs`) e só ganham um preço no instante exato em que são avaliados para cruzamento (via método `head()`) ou para a impressão do terminal. Isso atende ao edital com custo computacional nulo para reprecificação.

* **O `index` Global (O Roteador O(1)):**
  * **O Problema:** O edital introduz os comandos `cancel <id>` e `amend <id>`, fornecendo apenas a string do ID. Sem um índice global, teríamos que vasculhar todas as gavetas de todos os preços para achar a ordem.
  * **A Solução:** A `MatchingEngine` mantém um mapa espelho `id -> Order`. Quando o comando de alteração chega, o sistema encontra o objeto na memória instantaneamente, calcula em qual `BookSide` e em qual preço ele está, e opera a mutação in-loco.

## 2. A Arquitetura das Ordens "Pegged" (O Peg Virtual)

No jargão purista de mercado, "Pegged" é uma instrução de execução atrelada a uma ordem Limit, mas arquiteturalmente, tratá-la como um tipo isolado (`type = 'peg'`) eliminou uma árvore complexa e ineficiente de condicionais (`if/else` aninhados).

* **Passividade Estrutural e "Fila VIP":** Ordens Pegged são passivas por definição. Elas foram isoladas do loop de agressão e repousam em uma fila própria (`pegs`) em cada lado do livro.
* **O Peg Virtual `O(1)`:** Abandonamos a abordagem ingênua de "reprecificar fisicamente" os pegs a cada flutuação do mercado (o que destruiria a performance). O preço do Peg é derivado dinamicamente no momento do cruzamento (na função `head()`).
* **Prioridade Temporal Implacável:** No momento da execução, o motor compara o ID temporal (`seq`) da ordem Limit real mais antiga no topo do livro com o Peg mais antigo. Vence quem chegou primeiro.
* **Dormência de Pegs (Edge Case):** Se a liquidez real de um lado for totalmente consumida, os Pegs perdem sua âncora de preço. Em vez de cruzar o spread às cegas ou travar o motor, eles entram em hibernação. A função `head()` passa a retornar `None`, e os Pegs aguardam silenciosamente até que um formador de mercado real injete liquidez novamente no nível.

## 3. Segurança de Estado e Regras de Alteração (Amend)

A mutação de ordens em repouso é o vetor mais comum de bugs em exchanges. As seguintes travas foram implementadas:

* **A Trava Antirressurreição (`leaves_qty`):** A variável fundamental para alterações não é a quantidade original, mas sim a quantidade restante executável (`leaves_qty`).
  * **O Problema:** Se uma ordem de 100 cotas já executou 60 (restam 40), e o usuário pede um `amend` para 70, matematicamente 70 < 100 (o que manteria a prioridade). Porém, na prática, isso injetaria 30 cotas "fantasmas" no mercado furando a fila.
  * **A Solução:** O `amend` compara a nova quantidade estritamente com o saldo atual. Reduções mantêm a prioridade; aumentos (ou mudança de preço) perdem a prioridade, gerando um novo `seq` e movendo a ordem para o fim da fila.
* **O Escudo de Spread (`_cruzaria`):** Regras de mercado determinam que "amend nunca agride". Se um usuário alterar o preço de uma ordem de compra de 10 para 100 e já houver vendedores a 20, repousar a ordem a 100 corromperia o sistema criando um "livro cruzado" (bid >= ask sem trade). O escudo intercepta isso e lança um `ValueError`.
* **Transações Atômicas:** A validação ocorre estritamente antes da mutação. Se um `amend` for ilegal, o livro de ofertas permanece 100% intacto, sem que nenhuma ordem se perca no limbo da memória.

## 4. Precisão, Otimização e Execução

* **A Morte do Float:** A precisão financeira é inegociável. Para evitar as clássicas falhas de ponto flutuante na conversão binária, adotamos a biblioteca `Decimal` nativa, processando valores na base 10 e com sanitização canônica.
* **Otimização de Memória (`slots=True`):** A classe `Order` utiliza `@dataclass(slots=True)`. Ao bloquear a criação dinâmica do dicionário interno (`__dict__`) que o Python atribui a cada objeto por padrão, aproximamos a eficiência de memória da classe à de uma `struct` em linguagens de baixo nível. Isso poupa recursos significativos do Garbage Collector sob alta carga de ordens.
* **Relógio Determinístico (`seq`):** Para mitigar colisões de concorrência onde duas ordens chegam no mesmo milissegundo, abandonamos os carimbos de tempo (`timestamps`). O motor de prioridade usa um gerador determinístico inteiro (`itertools.count`).
* **Sweeping the Book:** Limit Orders agressivas (Marketable Limits) estão configuradas para varrer o livro adversário. Elas consomem múltiplos níveis de preço, recalculando o saldo continuamente até que o `leaves_qty` chegue a zero ou o preço limite estoure.
* **Consolidação de Trades:** Execuções parciais consecutivas consumindo múltiplas ordens no mesmo nível de preço são consolidadas em um único registro no log.

## 5. Auditoria de Invariantes e Política Fail-Fast

O motor foi construído sob o princípio Pythonico EAFP (Easier to Ask for Forgiveness than Permission).

* Em vez de sobrecarregar o fluxo crítico com múltiplos blocos condicionais de verificação, os métodos de remoção delegam a validação de existência diretamente para a implementação em C do interpretador, capturando o `KeyError` nativo em caso de violação.
* **Falha Dura (`RuntimeError`):** Se, por qualquer desvio matemático, o topo do vetor de preços apontar para uma gaveta vazia, o sistema é projetado para levantar um erro crítico imediato, preferindo interromper o fluxo a operar com base em um "estado fantasma".

### A Suíte de Testes (Pytest)

O repositório inclui uma bateria de 60 testes unitários automatizados cobrindo os cenários mais agressivos (FIFO misto, dormência de pegs, book varrido). O grande diferencial da suíte é o **Auditor Contínuo de Invariantes** (Pytest Fixture). Ao final de cada teste, o auditor congela a memória do motor e valida transversalmente se:
1. O livro não cruzou.
2. Não existem níveis de preço vazios ocupando memória.
3. As instâncias da matriz global `O(1)` correspondem exatamente aos objetos das gavetas (sem dessincronização).

---

## Como Executar

**Interface de Terminal (CLI):**
O motor pode ser alimentado diretamente via terminal (STDIN).

```bash
python matching_engine.py
```
*Exemplo de uso:*

```bash
> limit buy 10 100 order1
> peg bid buy 50 order2
> market sell 120
> print
```

**Rodando os Testes:**
Certifique-se de possuir o `pytest` instalado e execute na raiz do diretório:

```bash
pytest test_engine.py -v
```