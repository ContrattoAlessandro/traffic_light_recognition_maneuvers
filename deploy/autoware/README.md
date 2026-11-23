# R per Autoware e AWSIM: nodo ROS 2

Nodo ROS 2 che fa girare il modello R (RF-DETR Medium fattorizzato, input 1280, epoca 2, soglia 0.35) sulla camera
semafori di AWSIM e pubblica il colore della manovra dell'auto sul topic letto dal planner di Autoware. Il protocollo
di prova è in `KPI.md`.

È il nodo scritto e provato per F8, con il modello R e senza l'uscita U-turn. Il codice di inferenza ripete passo per
passo `deploy/reference_infer.py` (stesso resize Pillow, stesse classi, stessa regola). L'ONNX, le costanti e i frame di
verifica sono quelli del pacchetto hand-off (`deploy/README.md`).

**Stato: non ancora provato su ROS 2 reale, Autoware o AWSIM.** Vedi "Verifiche" in fondo.

## Come funziona

```
/sensing/camera/traffic_light/image_raw ──> R (ONNX) ──> colore per dritto, sinistra, destra
lanelet2_map.osm + percorso + posizione ──> prossima linea di stop e manovra della corsia
                                            └──> colore della manovra ──> TrafficLightGroupArray
```

- Il nodo gira solo entro 120 m dalla prossima linea di stop sul percorso.
- Pubblica un solo elemento circolare col colore della manovra, sull'id del regulatory element del semaforo, che è la
  chiave usata dal planner. Frecce e corsia sono già risolte nel nodo dalla regola di R.
- **Politica per `none`:** se non vede nessun semaforo della giunzione pubblica `UNKNOWN`, che il planner tratta come
  stop. Lo stesso vale per i semafori successivi del percorso, perché Autoware tratta come via libera un semaforo per
  cui non ha mai ricevuto nulla. È una scelta di sicurezza fissata prima delle prove: il prezzo è che l'auto può restare
  ferma se R non vede il semaforo dalla linea di stop.
- Il verde richiede 3 frame consecutivi; rosso e giallo agiscono subito. Il topic di debug riporta sempre anche il
  colore grezzo per frame (`raw_color`), quindi non serve ripetere le prove col filtro spento.
- La mappa serve solo a trovare la linea di stop, l'id del gruppo e `turn_direction`: mai a cercare o scegliere i
  semafori nell'immagine.

## Contenuto

| Percorso | Cosa contiene |
|---|---|
| `KPI.md` | Protocollo di prova ridotto: fasi, topic, KPI |
| `install.sh`, `setup_env.sh`, `requirements.txt` | Installazione offline e ambiente del nodo |
| `scripts/` | Avvio di Autoware e del nodo, controlli, self test |
| `ros2_ws/src/tl_maneuver_perception/` | Pacchetto ROS 2 (Python, nessuna dipendenza da torch o cv_bridge) |

La cartella di consegna (`E:\Autoware_TL_Maneuver_R`) contiene in più:

| Percorso | Cosa contiene |
|---|---|
| `model/` | `r_1280_opset17.onnx`, `model_config.json` e `parity.json` dal pacchetto hand-off |
| `golden/` | I 7 frame di verifica con le manovre attese, per il self test |
| `wheels/` | Runtime ONNX, CUDA e cuDNN per Linux, così non serve internet |

## Requisiti

- Ubuntu 22.04, ROS 2 Humble, Python 3.10, GPU NVIDIA con driver funzionante (`nvidia-smi`).
- Autoware compilato, con `TrafficLightGroupArray` (da giugno 2024 in poi).
- AWSIM o AWSIM Labs con la mappa Nishi-Shinjuku.
- Circa 5 GB liberi su disco. CUDA e cuDNN inclusi restano in questa cartella e non toccano quelli di Autoware.

## Installazione

Copia la cartella dalla chiavetta al disco Linux: `install.sh` si rifiuta di partire da una chiavetta. Poi:

```bash
cd ~/Autoware_TL_Maneuver_R
bash install.sh ~/autoware
```

Se Autoware non è in `~/autoware`, passa il percorso giusto. L'ultima riga attesa del self test è `self test passed`,
con `execution provider: CUDAExecutionProvider`.

## Avvio

Tre terminali, in quest'ordine, tutti nella cartella della consegna. `<MAP>` è la cartella con `lanelet2_map.osm` e
`pointcloud_map.pcd`, la stessa data ad Autoware.

1. **AWSIM.** Avvialo come al solito e verifica che la camera semafori sia accesa.
2. **Autoware:** `bash scripts/run_autoware.sh <modo> <MAP>`
3. **Nodo R:** `bash scripts/run_node.sh <modo> <MAP>`

Poi imposta posa iniziale e obiettivo in RViz e attiva la guida autonoma. Il nodo inizia a pubblicare appena riceve il
percorso.

| Obiettivo | Autoware | Nodo |
|---|---|---|
| Autoware guida, R osserva (Fasi 1-2) | `run_autoware.sh shadow <MAP>` | `run_node.sh shadow <MAP>` |
| R guida l'auto (Fasi 3-4) | `run_autoware.sh replace <MAP>` | `run_node.sh replace <MAP>` |

In `replace`, `run_autoware.sh` spegne la percezione semafori di Autoware con l'argomento di lancio che la versione
installata espone (`perception/enable_traffic_light` o `use_traffic_light_recognition`). Con AWSIM classico, prima di
`run_autoware.sh`: `export VEHICLE_MODEL=sample_vehicle SENSOR_MODEL=awsim_sensor_kit`.

## Controllo durante l'esecuzione

```bash
bash scripts/check_runtime.sh replace     # oppure shadow
source setup_env.sh && ros2 topic echo /tl_maneuver_perception/debug/decision
```

In `replace` il topic del planner deve avere **un solo publisher**, `tl_maneuver_perception`. Se ce ne sono altri, la
percezione semafori di Autoware è ancora attiva.

## Parametri

In `ros2_ws/src/tl_maneuver_perception/config/tl_maneuver.param.yaml`. Sono fissati prima delle prove e non vanno
ritoccati sui dati AWSIM.

| Parametro | Default | Significato |
|---|---|---|
| `threshold` | -1 (0.35) | Soglia sullo score, scelta sui tune split |
| `activation_distance` | 120 m | Distanza dalla linea entro cui il modello gira |
| `go_confirm_frames` | 3 | Frame verdi consecutivi per pubblicare verde |
| `miss_hold_frames` | 5 | Frame senza semaforo in cui il verde confermato resta valido |

## Verifiche

**Fatte su Windows:**

- Self test sui 7 frame golden (5 AWSIM, 1 DTLD, 1 OpenLane-V2), con onnxruntime su CPU: tutte le manovre attese.
- Logica di mappa, percorso, linea di stop, manovra, filtro e regola: `python3 test/test_logic.py` nella cartella del
  pacchetto, 5 test su una mappa sintetica.
- Per F8, con lo stesso codice: nodo completo con ROS e messaggi simulati, proiezione lat/lon verso MGRS contro pyproj.
- Parità ONNX contro il modello valutato: `model/parity.json` (99.3-100% delle manovre col resize Pillow).

**Da fare sulla macchina Linux:**

- Build `colcon`, avvio con ROS 2 reale, comunicazione con Autoware e AWSIM, CUDA con le librerie incluse.
- Lettura della mappa Nishi-Shinjuku vera (provata solo su una mappa sintetica).
- Latenza: la rete misurata in PyTorch bf16 impiega 29.5 ms; in ONNX float32 a 1280 su una GPU condivisa con AWSIM non
  è misurata. Il topic di debug riporta `processing_ms`. Un segnale più vecchio di 1 s vale come stop per il planner.

## Limiti

- **Non è un modello TensorRT.** I nodi TensorRT di Autoware accettano solo le loro architetture, quindi R gira in un
  nodo proprio con ONNX Runtime su CUDA. Un engine TensorRT si può aggiungere solo se la latenza non basta.
- **Semafori pedonali e V2I.** In `replace` la percezione semafori di Autoware è spenta, quindi il modulo strisce non
  riceve lo stato dei semafori pedonali e V2I non viene letto.
- **Scena giapponese.** Semafori orizzontali e lampade piccole sono i punti deboli di R (`deploy/AGENT_README.md`,
  sezione 7). R non ha mai visto immagini sintetiche.
- **Il percorso è necessario.** Senza percorso il nodo non pubblica nulla. Una corsia senza `turn_direction` vale
  "dritto". R non ha uscita U-turn.

## Problemi comuni

| Sintomo | Causa probabile |
|---|---|
| `execution provider: CPUExecutionProvider` | CUDA non caricato: controlla `nvidia-smi` e usa sempre `setup_env.sh` nel terminale del nodo |
| Nessuna uscita su `traffic_signals` | Manca il percorso o la posizione: `check_runtime.sh` |
| `lanelet ... is not in the map file` | Mappa diversa tra nodo e Autoware |
| Più publisher sul topic del planner | Autoware non è stato avviato con `replace` |
| L'auto non parte mai | Guarda `raw_color` nel topic di debug: se è `none`, il semaforo non viene visto |
| Errore sul tipo del messaggio | Autoware precedente a giugno 2024, serve un adattatore per `TrafficSignalArray` |
