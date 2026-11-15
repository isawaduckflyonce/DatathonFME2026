# Copilot Matemàtic Offline amb Arduino UNO Q i M3Kang

Aquest projecte implementa un **copilot matemàtic 100% offline** sobre la placa **Arduino UNO Q (4 GB)** per ajudar estudiants de primària i secundària a resoldre problemes de les proves Cangur ([M3Kang](https://huggingface.co/datasets/qualcomm/M3Kang)) sense que cap dada surti del dispositiu. El sistema combina **aprenentatge guiat** (escala de pistes de 4 nivells, detector de problemes "bessons" per practicar errors i feedback visual a la matriu LED de 8×13) amb **raonament automàtic** en anglès pas a pas dins el límit de 60 segons per problema.

---

## Què fa el nostre copilot

El copilot resol els dos reptes del Datathon FME 2026 (**Ensenyar** i **Raonar**) executant-se íntegrament en local:

1. **Mode Ensenyar (Interfície Web + Matriu LED 8×13):**
* **Escala de pistes progressiva (`copilot/hints.py`):** Quan un estudiant s'encalla, mai li dona la resposta de cop, sinó que puja quatre esglaons:


1. *Nivell 1 (Exemple més fàcil):* Cerca als *embeddings* de M3Kang un problema resolt del mateix idioma i tema, però d'un nivell de dificultat estrictament inferior.


2. *Nivell 2 (Tema i estratègia):* Classifica el problema en un dels 6 grans blocs temàtics (Geometria, Combinatòria, Lògica, Teoria de nombres, Àlgebra/Ràtios o Patrons) a partir del vocabulari bilingüe (català/anglès) i dels seus 5 veïns més propers, i suggereix una heurística de resolució.


3. *Nivell 3 (Pregunta guia socràtica):* Genera una única pregunta guia en l'idioma de l'alumne (< 20 paraules) amb un filtre anti-filtracions (`contains_answer_leak`) que bloqueja automàticament tant la resposta correcta com qualsevol número nou calculat que no aparegués a l'enunciat.


4. *Nivell 4 (Solució explicada):* Només quan l'alumne ho demana explícitament, mostra la lletra correcta i una explicació breu en 2–3 frases.




* **Recomanador de "Bessons d'Errors" (`explorer/server.py`):** Si l'estudiant falla un problema, el servidor calcula un perfil TF-IDF reforçat amb els veïns semàntics de l'índex per proposar un problema nou ("bessó") del mateix tema, idioma i dificultat per consolidar el concepte.


* **Feedback físic a la matriu LED (`copilot/matrix.py` + `sketch/sketch.ino`):** Cada vegada que l'usuari respon des del navegador, la placa dibuixa instantàniament un tic (`✓`) si l'ha encertat o una creu (`✗`) si ha fallat a la matriu de 8×13 LEDs integrada.




2. **Mode Raonar (`predict.py` + `copilot/pipeline.py` + `copilot/llm.py`):**
* Quan s'avalua de manera autònoma, força el model `Qwen3.5 0.8B` a pensar i escriure **estrictament en anglès** mitjançant una gramàtica GBNF amb 2 a 5 passos curts de càlcul (`Steps:`), un resultat numèric (`Result:`) i un aparellament explícit amb una opció (`Match: ... option <A-E>`), seguit de la verificació del valor amb les opcions OCR.


---

## Arquitectura del sistema

La placa **Arduino UNO Q (4 GB)** divideix la càrrega de treball entre el processador Linux (Qualcomm QRB2210 de 4 nuclis Arm) i el microcontrolador en temps real:

```text
+-----------------------------------------------------------------------------------+
|                         ARDUINO UNO Q (4 GB RAM - 100% OFFLINE)                  |
|                                                                                   |
|  +---------------------------------------------+   +---------------------------+  |
|  |       LINUX (Qualcomm QRB2210 Arm)          |   |     MICROCONTROLADOR      |  |
|  |                                             |   |                           |  |
|  |  [systemd] math-llm.service (:8081)         |   |  sketch/sketch.ino        |  |
|  |   └─ llama-server: Qwen3.5-0.8B-Q8_0        |   |   ├─ Arduino_LED_Matrix   |  |
|  |                                             |   |   │   (8x13 = 104 LEDs)   |  |
|  |  [systemd] math-embed.service (:8082)       |   |   └─ Arduino_RouterBridge |  |
|  |   └─ llama-server: bge-small-en-v1.5-q8_0   |   |       show_result(state)  |  |
|  |                                             |   +-------------^-------------+  |
|  |  [Python stdlib]                            |                 |                |
|  |   ├─ explorer/server.py (:8000) + index.html|                 | MessagePack    |
|  |   ├─ copilot/pipeline.py & copilot/llm.py   |                 | RPC over Unix  |
|  |   ├─ copilot/hints.py (Escala 4 nivells)    |                 | Socket         |
|  |   ├─ copilot/matrix.py ─────────────────────+-----------------+                |
|  |   └─ data/index.db (SQLite + vectors M3Kang)|  (/var/run/arduino-router.sock)  |
|  +---------------------------------------------+                                  |
+-----------------------------------------------------------------------------------+

```

* **Què s'executa a Linux:**
* Dos serveis `systemd` permanents basats en `llama.cpp`: `math-llm.service` (port `8081`, context `4096`, `--cache-ram 0`, `--reasoning off`) i `math-embed.service` (port `8082`, `--embedding`, vectors de 384 dimensions).


* El codi de control en **Python pur (només llibreria estàndard, sense `pip` ni dependències pesades)**: `copilot/pipeline.py` (orquestració dins el límit de 60 s), `copilot/llm.py` (prompting en anglès amb memòria cau KV i restriccions GBNF), `copilot/hints.py` (pistes de 4 nivells i detector anti-filtracions), `copilot/matrix.py` i `explorer/server.py` (servidor HTTP al port `8000`).




* **Què s'executa al microcontrolador:**
* L'*sketch* `sketch/sketch.ino` inicialitza la matriu `Arduino_LED_Matrix` i exposa la funció RPC `show_result(int state)` a través d'`Arduino_RouterBridge` (`1` = tic verd/encès, `2` = creu, `0` = neteja). Linux s'hi comunica de forma asíncrona enviant paquets MessagePack-RPC al sòcol `/var/run/arduino-router.sock`.




* **Com utilitzem M3Kang:**
* Els problemes resolts del conjunt d'entrenament s'indexen en local a `data/index.db` (SQLite) amb els seus *embeddings*. Aquesta base de coneixement alimenta alhora el RAG (`--rag`), la cerca d'exemples més fàcils (Nivell 1), la classificació temàtica per veïns (Nivell 2) i la selecció de problemes bessons quan l'estudiant s'equivoqui.





---

## Com replicar-ho pas a pas

Partint d'una placa Arduino UNO Q configurada amb els **passos 0 a 12** del repositori base (serveis `math-llm` i `math-embed` actius, dades `data/train.jsonl` i `data/val.jsonl` copiades i índex `data/index.db` generat):

1. **Clonar el nostre repositori a la placa:**
```bash
cd ~
git clone <URL_DEL_VOSTRE_REPOSITORI> datathon-UPC-maths-qualcomm-arduino-unoq
cd datathon-UPC-maths-qualcomm-arduino-unoq

```


2. **Carregar l'sketch al microcontrolador (per a la matriu LED 8×13):**
* Obre el projecte a **Arduino App Lab** (o compila `sketch/sketch.ino` contra la placa UNO Q) perquè el microcontrolador registri el servei `show_result` al *router bridge*.




3. **Comprovar que els dos models locals responen:**
```bash
systemctl is-active math-llm math-embed
curl -s http://127.0.0.1:8081/health; echo
curl -s http://127.0.0.1:8082/health; echo

```


(Ambdues comprovacions han de retornar `active` i `{"status":"ok"}`.)


4. **Executar la verificació ràpida (smoke test) i generar prediccions:**
```bash
mkdir -p results
python3 predict.py --input data/sample_problems.jsonl --output results/smoke.csv
cat results/smoke.csv

```


5. **Llançar l'explorador web interactiu amb pistes i matriu LED:**
```bash
python3 explorer/server.py --host 0.0.0.0 --port 8000

```


* Per obrir-lo des del portàtil per USB (sense Wi-Fi, tal com requereix la demo):



```bash
adb forward tcp:8000 tcp:8000

```

* Obre `[http://127.0.0.1:8000](http://127.0.0.1:8000)` al navegador: respon amb `A`–`E` per veure el tic o la creu a la placa, demana pistes progressives (nivells 1 a 4) o practica amb el problema bessó en cas d'error.



---

## Resultats

> **Nota sobre l'avaluació:** Degut al temps d'execució a la placa (~46,7 s per problema) i a la manca de temps abans del lliurament final, **no ens ha estat possible completar una avaluació exhaustiva sobre tot el conjunt de validació ni desar la taula completa a `results/ablations.csv**`. A continuació es presenten els resultats preliminars mesurats sobre els **25 primers problemes** del conjunt de prova (`eval.py`), comparats amb el *baseline* oficial del repositori:
> 
> 

| Experiment / Mètode | Problemes avaluats | Encert (Accuracy) | Latència mitjana | Pitjor temps | Timeouts (> 60 s) |
| --- | --- | --- | --- | --- | --- |
| Atzar (5 opcions A–E) | — | 20,0% | 0,0 s | 0,0 s | 0 |
| *Baseline* oficial (1 lletra, gramàtica `[A-E]`)

 | 40 (eng) | 22,5% | 8,7 s | 12,7 s | 0 |
| *Baseline* 1 línia de raonament + lletra

 | 40 (eng) | 17,5% | 26,3 s | 35,0 s | 0 |
| **El nostre copilot (CoT en anglès + `Result` + `Match`)** | **25 (eng + cat)** | **32,0% (8/25)** | **46,7 s** | **57,0 s** | **0** |

* **Millora sobre el *baseline*:** Tot i ser una mostra parcial de 25 problemes (14 en anglès i 11 en català, barrejant tots els nivells de dificultat del 0 al 7), l'encert puja fins al **32,0%** (+9,5 punts percentuals respecte al 22,5% del *baseline* en anglès i +14,5 punts respecte al raonament d'una línia del repositori base), mantenint **0 timeouts** per sobre del límit de 60 segons.



---

## Anàlisi d'errors (per idioma, dificultat i tema)

A partir dels 25 problemes registrats a l'execució preliminar, s'observen els patrons següents:

### 1. Per idioma

| Idioma | Encerts | Percentatge | Observacions |
| --- | --- | --- | --- |
| **Català (`cat`)** | 4 / 11 | **36,4%** | Forçar el *system prompt* i la gramàtica GBNF a escriure en anglès des del primer token elimina la pèrdua de temps traduint l'enunciat i permet que el model raoni amb el seu idioma més fort. |
| **Anglès (`eng`)** | 4 / 14 | **28,6%** | Rendiment estable, penalitzat principalment pels problemes geomètrics de nivells alts sense suport visual. |

### 2. Per nivell de dificultat (`level` 0 a 7)

| Nivell | Encerts | Percentatge | Nivell | Encerts | Percentatge |
| --- | --- | --- | --- | --- | --- |
| **Nivell 0** | 1 / 4 | 25,0% | **Nivell 4** | 1 / 1 | 100,0% |
| **Nivell 1** | 1 / 2 | 50,0% | **Nivell 5** | 2 / 5 | 40,0% |
| **Nivell 2** | 0 / 5 | 0,0% | **Nivell 6** | 0 / 1 | 0,0% |
| **Nivell 3** | 1 / 4 | 25,0% | **Nivell 7** | 2 / 3 | 66,7% |

* **Distribució de lletres (predites / reals):** `A: 15/6`, `B: 2/5`, `C: 3/4`, `D: 4/7`, `E: 1/3`. Encara persisteix un cert biaix cap a la lletra `A` quan el valor calculat a `Result:` no coincideix exactament amb cap opció i `pipeline.py` executa la segona crida de verificació (`cot_miss_calibrated`), fet que també explica per què el temps mitjà arriba als 46,7 s (dues crides consecutives al model en lloc d'una).



### 3. Per tema

* El conjunt de dades M3Kang no inclou etiquetes de tema explícites (els 6 temes de `copilot/hints.py` s'infereixen dinàmicament per paraules clau) i per manca de temps no s'ha pogut generar un desglossament quantitatiu per categoria.


* Qualitativament, el sistema funciona millor en **problemes aritmètics, de ràtios i de recompte directe** enunciats en text, mentre que els principals errors es concentren en **geometria espacial i rotacions**, on l'enunciat depèn d'una figura que el model de text no veu.



---

## Què no ha funcionat

Durant el desenvolupament vam provar i descartar tres aproximacions abans d'arribar a l'arquitectura actual:

1. **Generació pura d'equacions simbòliques (`var = expr`) amb resolutor extern AST:**
* Vam intentar que `Qwen3.5 0.8B` emetés únicament assignacions de variables (`diff = 17 - 9`, `ans = diff / 2`) per resoldre-les amb un *parser* matemàtic en Python. Un model de 0,8B paràmetres no té prou capacitat d'abstracció per formalitzar enunciats del Cangur en variables consistents: sovint inventava nombres inexistents, entrava en bucles de decimals (`10.10000000...`) o esgotava els tokens abans d'arribar a la resposta.


2. **Frases d'un sol vers ultracurtes (`max_tokens=32`):**
* Limitar el raonament a una sola línia de ≤ 55 caràcters feia que el model gastés tot el pressupost de tokens parafrasejant l'enunciat i quedés tallat just abans de fer el càlcul (`...the total change is\nAnswer:`), provocant un col·lapse gairebé total cap a la lletra `A` (29/30 prediccions).


3. **Permetre generació lliure en català sense restricció GBNF:**
* Sense una gramàtica GBNF ASCII-only que forci l'inici en anglès, el model dedicava gran part dels tokens (a ~5,6 tokens/s) a repetir o traduir l'enunciat català, exhaurint el temps disponible sense arribar a cap conclusió.





---

## Membres de l'equip

* **Adrià González Roigé**
* **Alan Mena Comaleras**
* **Roger Fernández Lopera**

---

## Llicència i citació de dades

* **Codi del projecte:** Publicat sota llicència **MIT**.
* **Conjunt de dades M3Kang:** Utilitzat exclusivament per a recerca sense redistribuir cap fitxer de dades (`data/`, `*.db`, `*.gguf`) al repositori. Citació: *Torres-Camps et al., 2026, M3Kang ([arXiv:2601.16218](https://arxiv.org/abs/2601.16218)), Qualcomm*.
