# Protocollo di prova di R in Autoware e AWSIM (ridotto)

**Domanda.** Se la ricerca del semaforo guidata dalla HD map viene sostituita da R sulla sola immagine, l'auto
rispetta i semafori come prima, e dove sbaglia?

È un capitolo della tesi, non un articolo di valutazione: c'è una sola mappa e un solo simulatore, e ci sono decine
di avvicinamenti, non centinaia. Per questo il protocollo riporta conteggi e casi, non prove di non-inferiorità.
Sostituisce il `KPI.md` di F8 (7 gruppi di KPI, 3 bracci, perturbazioni della mappa).

## Fissato prima di guardare i risultati

| Scelta | Valore |
|---|---|
| Modello | R, epoca 2, soglia 0.35 (scelte sui tune split, non si ritoccano su AWSIM) |
| Regola | Regola di giunzione (convenzione tedesca), come nel resto della tesi |
| `none` | Pubblicato come `UNKNOWN` (stop) |
| Filtro | `go_confirm_frames: 3`, `miss_hold_frames: 5`. Il colore grezzo si legge comunque dal topic di debug |
| Baseline | La percezione nativa di Autoware nella configurazione di default di `e2e_simulator.launch.xml` |
| Stop / go | Come negli esperimenti: solo il rosso vale stop. False-Go = frame rossi predetti verde o giallo / frame rossi |

## Fasi

### Fasi 1-2: Autoware guida, R osserva (shadow)

Un'unica sessione fa sia la baseline sia la validazione di R sugli stessi frame:

```bash
bash scripts/run_autoware.sh shadow <MAP>
bash scripts/run_node.sh shadow <MAP>
```

1. Un percorso fisso su Nishi-Shinjuku, con più incroci semaforizzati, possibilmente almeno una svolta. Lo stesso
   percorso si usa in tutte le fasi.
2. Controlla che l'auto si fermi col rosso con la percezione nativa e che `check_runtime.sh shadow` mostri le
   decisioni di R.
3. Registra il percorso 2-3 volte (10-15 minuti in tutto) con `ros2 bag record` sui topic sotto.

Se la GPU non regge AWSIM, Autoware e R insieme, registra senza il nodo e fai girare R dopo sul rosbag. Avvia prima
il nodo in `shadow`, poi `ros2 bag play <bag> --clock`, così il nodo riceve il percorso.

### Fasi 3-4: R guida (replace)

```bash
bash scripts/run_autoware.sh replace <MAP>
bash scripts/run_node.sh replace <MAP>
bash scripts/check_runtime.sh replace      # un solo publisher sul topic del planner
```

Stesso percorso, 3-5 giri, stessi topic registrati.

### Topic da registrare

| Dato | Topic |
|---|---|
| Decisione usata dal planner (nativa in shadow, R in replace) | `/perception/traffic_light_recognition/traffic_signals` |
| Decisione di R in shadow | `/perception/traffic_light_recognition/maneuver/traffic_signals` |
| Dettagli di R per frame (colore grezzo e filtrato, distanza dalla linea, `processing_ms`) | `/tl_maneuver_perception/debug/decision` |
| Ground truth | Il topic dello stato vero dei semafori di AWSIM, se esiste: cercalo con `ros2 topic list \| grep -i -E 'v2x\|v2i\|traffic'` |
| Posizione e velocità | `/localization/kinematic_state` |
| Percorso | `/planning/mission_planning/route` |
| Immagine | `/sensing/camera/traffic_light/image_raw`, almeno nelle Fasi 1-2, per rivedere gli errori |

**Senza topic di ground truth** si etichetta a mano, per avvicinamento, lo stato vero del semaforo della manovra
come intervalli di tempo (rosso dalle t1 alle t2...) guardando le immagini del rosbag. Le fasi cambiano poche volte
per avvicinamento, quindi sono decine di etichette, non migliaia.

## KPI

**Avvicinamento:** un passaggio dell'auto verso una linea di stop semaforizzata, dai 120 m prima fino al superamento.
I frame consecutivi sono correlati: i KPI per frame si riportano insieme al numero di avvicinamenti.

### Shadow, sugli stessi frame (Fasi 1-2): R contro la percezione nativa

| KPI | Definizione |
|---|---|
| **False-Go** | Frame con rosso vero e verde o giallo predetto / frame con rosso vero. Il KPI principale |
| **False-Stop** | Frame con verde vero e rosso predetto / frame con verde vero |
| **Copertura** | Frame con un colore noto / frame valutati |
| **Distanza di aggancio** | Per ogni avvicinamento col rosso, la distanza dalla linea da cui la decisione di R resta corretta |
| **Matrice di confusione** | Colore vero contro colore predetto (più `none`), per R e per la nativa |

R si valuta col colore grezzo (`raw_color`), confrontabile con gli altri capitoli. Per la nativa vale il colore
che il planner riceve per il gruppo dell'avvicinamento.

### Closed-loop (Fasi 3-4): R guida

Una tabella di conteggi, accanto agli stessi numeri della nativa nelle Fasi 1-2:

| KPI | Definizione |
|---|---|
| **Passaggi col rosso** | Avvicinamenti in cui l'auto supera la linea col rosso, su avvicinamenti col rosso alla linea |
| **Arresti inutili** | Arresti di oltre 1 s col verde per la manovra dell'auto |
| **Blocchi** | Avvicinamenti in cui l'auto resta ferma per un ciclo intero dopo il verde |
| **Percorso completato** | Giri completati senza intervento |
| **Latenza** | `processing_ms` mediana e p95; segnali più vecchi di 1 s (`tl_state_timeout`) |

Ogni passaggio col rosso e ogni episodio di False-Go si analizza a parte, con l'immagine: semaforo non visto,
colore sbagliato, semaforo di un'altra giunzione, ritardo del filtro.

### Statistica

Conteggi con il loro denominatore, nessun test di ipotesi. Con zero passaggi col rosso su N avvicinamenti col rosso,
il limite superiore al 95% è circa 3/N, e va scritto: con 20 avvicinamenti il risultato è compatibile con fino al 15%.

## Facoltativo, solo se avanza tempo

Una perturbazione della mappa, l'esperimento che mostra meglio il vantaggio dell'approccio senza ROI. Una copia della
mappa con i semafori (le way `traffic_light` dei regulatory element) spostati di 2 m, data ad Autoware per un giro in
più delle Fasi 1-2; le linee di stop restano dove sono, quindi il planner non cambia. Al nodo si dà la mappa
originale: R non usa la posizione dei semafori, e si confronta il False-Go della nativa con e senza spostamento.

## Fuori dal protocollo, di proposito

Braccio oracolo in closed-loop (l'oracolo serve solo come etichetta), seconda configurazione della baseline col fine
detector, prove col filtro spento, braccio F7, comfort e jerk, precisione di arresto, VRAM e real-time factor,
perturbazioni di localizzazione e calibrazione, ritaratura della soglia su AWSIM, engine TensorRT (salvo latenza
insufficiente), frecce pubblicate come `LEFT_ARROW` / `RIGHT_ARROW`.
