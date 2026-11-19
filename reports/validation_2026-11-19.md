# Validation del 19 novembre 2026

Confronto eseguito localmente sui file reali del Drive condiviso. Il run completo di Colab resta da confermare: il controllo del browser non ha consentito l’esecuzione UI.

## Dati e protocollo

- Estrazione experiments completata su **26,182 film**: **24,518 trame**, **1,664 fallback**; copertura **93.64%** e nessun errore da ritentare.
- Lingue osservate: 17. Provenienza, revisione, lingua e stato restano in `plots.parquet`; i casi senza trama sono elencati in `plots_fallback.parquet`.
- Stesso campione di 2.000 utenti per tutti i modelli; accuratezza sui 1.958 con almeno un positivo in validation. I restanti 42 partecipano alle metriche oltre l’accuratezza.
- K=10, rilevanza rating >=4, catalogo di 22.280 film con almeno 10 rating di train; ogni film già valutato in train è escluso. Tutti i positivi validation entrano nel denominatore.
- Novelty e long-tail sono ricavati solo dal train. Diversity è la distanza Jaccard sui generi. Nessuna valutazione finale sul test è stata eseguita.

## Risultati

| Metrica | Popularity | Matrix Factorization | Content-Based |
| --- | ---: | ---: | ---: |
| Recall@10 | 0.057794 | 0.093520 | 0.003420 |
| NDCG@10 | 0.067500 | 0.101393 | 0.003603 |
| Novelty@10 (bits) | 1.681342 | 3.081354 | 9.416919 |
| Diversity@10 (genres) | 0.782454 | 0.767381 | 0.770576 |
| Long-tail share@10 | 0.000000 | 0.000000 | 0.829600 |
| Catalog coverage | 0.004533 | 0.039497 | 0.061580 |

## Content-Based: copertura e controlli

- Sentence Transformer multilingue fissato a modello e revisione; finestre di token per le trame lunghe, profilo medio dei film apprezzati e ranking coseno.
- Disponibilità delle trame di profilo limitata al catalogo di fitting. Film apprezzati fuori catalogo usano generi/tag, indipendentemente dai conteggi train+val.
- I tag precedono strettamente l’ultimo rating della finestra ammessa del proprio autore. Il test, quando sarà autorizzato, richiede la configurazione salvata in validation e usa train+val.
- Tutti i 2.000 utenti ricevono 10 raccomandazioni: 20.000 righe. Tre utenti senza profilo positivo utilizzabile ricevono il fallback Popularity, registrato nei diagnostici.
- Nel catalogo validation: 21.125 film con trama, 1.132 con generi/tag e 23 senza testo utilizzabile. Questi ultimi sono esclusi dal ranking semantico.
- Controllati schema, ID, revisioni, lunghezze, completezza dell’export, esclusione dei film già visti e allineamento delle 22.628 righe di embedding. Le 41 righe senza testo hanno vettore nullo; tutte le altre sono normalizzate.
- Verifica semantica del profilo Interstellar / The Martian / Arrival: tra i primi risultati compaiono Red Planet, 2001: A Space Odyssey, Space Cowboys e Apollo 13.
- Gli embedding sono stati precalcolati mentre proseguiva l’estrazione. Il tempo nel manifest misura il run con la cache disponibile.

## Limiti e prossima decisione

Il Content-Based è una prima baseline con accuratezza bassa e forte esposizione long-tail. Le raccomandazioni sono concentrate su alcuni titoli: Alone in the Dark (1982) compare per 1.066 utenti (53,3%). Questo risultato richiede analisi degli errori e tuning sulla validation con Andrea prima del test finale.

1. Review e merge della PR #8 (Evaluation), poi della PR #9 (trame e Content-Based).
2. Confermare in Colab `07_comparison.ipynb` sul branch di lavoro; dopo i merge usare `main`.
3. Concordare il miglioramento del Content-Based sulla validation; mantenere separato il test finale.

Gli output sono in `data/` e `results/` e vengono mantenuti fuori da Git. Il manifest `content_val_params.json` registra protocollo e versioni dell’encoder.
