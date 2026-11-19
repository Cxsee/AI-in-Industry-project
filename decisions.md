# Decisioni di progetto

Aggiornato il 13 novembre 2026. Per chiudere una voce, annotare la scelta, la data e il link alla discussione o alla PR.

## Prima di fissare dati e trame

- [ ] **Catalogo candidato e perimetro dell'estrazione.** Confermare `MIN_RATINGS = 10` sui rating di train + validation, oppure scegliere una soglia diversa. Con 10 rating ci sono 26.182 candidati e il 99,2% dei rating positivi del test riguarda quei film. La percentuale è una diagnostica aggregata della PR, quindi per scegliere la soglia usare la validation. Concordare se eseguire `01b_plots` in modalità `experiments` con `experiment_movies.parquet` o in modalità `catalog` su tutti gli 87.585 film. Il notebook su `main` parte in modalità `pilot`. [PR #3](https://github.com/Cxsee/AI-in-Industry-project/pull/3), [PR #4](https://github.com/Cxsee/AI-in-Industry-project/pull/4)
- [ ] **Split temporale.** Ratificare il protocollo della pipeline. Per ogni utente, l'ultimo 20% va al test e l'ultimo 20% del resto va alla validation, con arrotondamento per eccesso. Con cutoff per utente, i rating di train di un utente possono essere successivi ai rating di test di un altro. Dichiarare questa proprietà nella presentazione o adottare un cutoff temporale globale. Durante il tuning, calcolare dai soli dati di train le statistiche che influenzano i risultati sulla validation. Train + validation è la finestra attuale di `movie_stats.parquet` per il confronto finale sul test. [PR #4](https://github.com/Cxsee/AI-in-Industry-project/pull/4)
- [x] **Fonte delle trame.** Wikipedia tramite Wikidata: Francesco ha confermato il 18 novembre 2026 che la scelta è già concordata con Andrea. Allineare il testo della proposta iniziale, che indicava TMDB. [PR #2](https://github.com/Cxsee/AI-in-Industry-project/pull/2), [PR #3](https://github.com/Cxsee/AI-in-Industry-project/pull/3)
- [ ] **Trame multilingue e dati mancanti.** Scegliere tra embedding multilingue sulle trame originali e traduzione verso una lingua comune. Confermare quando usare il fallback a generi e tag previsto da `04_content_based` e come riportarne la copertura. [Review PR #3](https://github.com/Cxsee/AI-in-Industry-project/pull/3#pullrequestreview-5841315864)
- [ ] **Controllo qualità delle trame.** Decidere come campionare e valutare l'output automatico, includendo trame brevi e multilingue, e se limitare al pilot le correzioni manuali per singolo film. Il 73% di copertura è stato osservato su 100 film. [Review PR #3](https://github.com/Cxsee/AI-in-Industry-project/pull/3#pullrequestreview-5841315864)
- [ ] **Tag e leakage.** `tags.parquet` conserva tutti i tag. Definire il cutoff da applicare quando i tag entrano nel profilo utente o negli attributi dei film, escludendo dal modello i tag futuri rispetto alla finestra ammessa per la valutazione. [PR #4](https://github.com/Cxsee/AI-in-Industry-project/pull/4)

## Prima di confrontare i modelli

- [ ] **Formato delle raccomandazioni e utenti.** Concordare lo schema condiviso (proposta `userId`, `movieId`, `rank`, `score` quando disponibile), come escludere i film già valutati nella finestra di storia e come trattare ranking corti o mancanti. Confermare i 2.000 utenti di `users_sample.parquet` o un sottoinsieme comune per tutti i modelli se il costo LLM lo richiede. [PR #4](https://github.com/Cxsee/AI-in-Industry-project/pull/4), [Verifica progetto data](https://chatgpt.com/c/6af355b9-3138-83ed-ba5a-a0a0658d0870)
- [ ] **Protocollo del confronto finale.** Ratificare con il gruppo le convenzioni della validation e la loro applicazione al test. Decidere se affiancare al Recall una vista limitata ai positivi raccomandabili. Prima del confronto tra modelli, fissare il protocollo degli intervalli bootstrap al 95%, inclusi il ricampionamento degli stessi utenti per tutti i modelli e il trattamento della coverage globale. [PR #4](https://github.com/Cxsee/AI-in-Industry-project/pull/4), [Verifica progetto data](https://chatgpt.com/c/6af355b9-3138-83ed-ba5a-a0a0658d0870)
- [ ] **Diversity ed exposure.** Decidere se riportare entrambe le misure di diversity (Jaccard sui generi e distanza coseno sugli embedding) e quale usare come misura principale. Concordare un'unica rappresentazione per il confronto e la gestione di generi o embedding mancanti. Usare un catalogo comune come denominatore della coverage e confermare il confine long-tail (l'head attuale contiene i film che raggiungono l'80% dei rating). [PR #4](https://github.com/Cxsee/AI-in-Industry-project/pull/4), [Verifica progetto data](https://chatgpt.com/c/6af355b9-3138-83ed-ba5a-a0a0658d0870)
- [ ] **Discovery, cost e analisi LLM.** Fissare coorte e obiettivi di unfamiliar-interest per l'estensione MovieLens 32M del SIGIR 2025. Definire quali operazioni conteggiare nella latenza e come raccogliere token di input e output per utente. Per i tre ordini dei candidati previsti da `05_llm_ranker`, specificare come aggregare Kendall τ e trattare ranking incompleti o ex aequo. Concordare le fasce di popolarità e anno di uscita e l'analisi della posizione nel prompt. [Verifica progetto data](https://chatgpt.com/c/6af355b9-3138-83ed-ba5a-a0a0658d0870)

## Convenzioni già implementate

- `metrics.py` riceve liste di `movieId` in ordine di rank, con `K = 10` di default. Rilevanza e catalogo sono forniti dal chiamante. Recall usa tutti i positivi forniti e NDCG è binario con ranking ideale fino a K. Senza positivi il punteggio individuale di accuratezza è zero. Le medie di lista usano la lunghezza effettiva. ID duplicati nel prefisso valutato, film fuori catalogo nella coverage e metadati richiesti mancanti generano un errore.
- La diversity è la media delle distanze tra coppie distinte e vale zero con meno di due film. La coverage conta i titoli distinti raccomandati a tutta la coorte.
- La correzione del setup Colab della PR #1 è stata propagata ai notebook della PR #2. La cella è duplicata. Quando servirà aggiornarla, decidere se continuare a sincronizzare le copie o centralizzare il setup. Il notebook delle trame ha già recepito setup, percorsi, dipendenze e pulizia degli output richiesti in review. [PR #1](https://github.com/Cxsee/AI-in-Industry-project/pull/1), [PR #2](https://github.com/Cxsee/AI-in-Industry-project/pull/2), [PR #3](https://github.com/Cxsee/AI-in-Industry-project/pull/3)

### Prima valutazione sulla validation

- **Coorte comune.** Si parte dai 2.000 utenti di `users_sample.parquet`. Per Recall e NDCG si usano solo gli utenti con almeno un rating `val >= 4.0`. I punteggi di accuratezza degli altri utenti sono nulli e il riepilogo riporta quanti utenti sono stati esclusi. Il filtro dipende dalle etichette della validation e si applica allo stesso modo a ogni modello.
- **Rilevanza e cutoff.** La rilevanza è binaria e il cutoff è 10. Il denominatore del Recall include tutti i film positivi in validation, anche quelli fuori dal catalogo ammissibile. NDCG usa lo stesso insieme di positivi per il ranking ideale.
- **Catalogo e statistiche.** Solo i rating di `train` determinano il catalogo (almeno 10 rating per film), la storia già vista, la novelty e il long-tail. Per la novelty, `p(i)` è il numero di rating del film diviso per il numero di utenti distinti nel train. L'head è calcolato sull'intera distribuzione dei rating di train, prima di filtrare il catalogo. `movie_stats.parquet` usa train + validation ed è riservato al confronto finale.
- **Ranking e aggregazione.** Si accettano liste di massimo 10 film distinti, con rank consecutivi da 1, senza film già valutati in train o fuori catalogo. Le liste mancanti sono vuote e ricevono zero nelle metriche applicabili. Recall e NDCG sono medie per utente della coorte di accuratezza. Novelty, diversity sui generi e long-tail share sono medie per utente sull'intero campione. La coverage divide i film distinti consigliati a tutto il campione per la dimensione del catalogo di train. I risultati si salvano in `popularity_val_user_metrics.parquet` e `popularity_val_summary.parquet`.

## Content-Based: protocollo implementato il 18 novembre 2026

- Validation: solo train per candidati (almeno 10 rating), profili, esclusione dei film già
  valutati, popolarità di fallback e cutoff dei tag. Test: train + val con configurazione
  salvata in validation. I tag devono precedere strettamente l'ultimo rating ammesso del
  proprio autore; autori senza storia e tag al confine sono esclusi. Resta il protocollo
  temporale per utente già usato dagli altri modelli, senza cutoff globale.
- Embedding multilingue delle trame accettate; fallback a generi e massimo 20 tag disponibili.
  Modello e revisione sono fissati nel notebook. Le finestre di token delle trame lunghe
  vengono aggregate prima di costruire il profilo medio dei film con rating >= 4.
- Ranking coseno, esclusione di tutti i film già valutati, pareggi per movieId crescente.
  Film senza alcun testo esclusi dal ranking semantico. Utenti senza profilo positivo
  utilizzabile: fallback Popularity della sola finestra ammessa, riportato separatamente.
- Queste sono scelte implementative da controllare nella PR, non risultati sperimentali.
  La copertura finale viene dal run completo: non estendere il 73% del pilot al catalogo.

- Disponibilità delle trame nei profili: si usano embedding delle trame solo per film
  nel catalogo della finestra di fitting. Per i film apprezzati fuori da quel catalogo
  si applica sempre il fallback generi/tag, anche se una trama è presente nell'export
  experiments costruito da train + val. In questo modo la disponibilità delle feature
  in validation non dipende dai conteggi futuri usati per estrarre il catalogo completo.
