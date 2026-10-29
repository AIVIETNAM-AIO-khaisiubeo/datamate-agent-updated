# data/

Put the official split here (layout is auto-discovered, folders searched recursively):

```
data/<split>/
  questions.jsonl              {"id","business_id","question","as_of_date"}
  rulebook.*                   shared rulebook (md/txt/yaml/json) — auto-discovered by name
  *lunar*.csv | *calendar*     shared solar–lunar calendar — auto-discovered by name
  <business_id>/               one folder per business: *.csv, *.xlsx, *.sqlite, README.md
```

`make synth` writes a synthetic split with the same layout plus `gold.jsonl`.
