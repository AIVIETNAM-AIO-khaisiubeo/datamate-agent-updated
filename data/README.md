# Data directory

Place the official competition split here. The loader searches recursively for question metadata, the rulebook, calendar files, and business data.

~~~text
data/<split>/
├── questions.jsonl             # id, business_id, question, as_of_date
├── rulebook.md                 # or another supported rulebook text/config file
├── calendar.csv                # optional solar–lunar calendar
└── <business_id>/
    ├── *.csv
    ├── *.xlsx
    ├── *.sqlite
    └── README.md               # optional column dictionary and business conventions
~~~

Each business folder may contain multiple files. The loader uses the shop README as evidence for column roles and business conventions, but does not assume every description is complete or correct.

The rulebook and calendar are shared split-level inputs. The rulebook is treated as the authoritative source of business rules; the calendar file takes precedence over the built-in lunar-calendar fallback when present.

The command make synth creates a synthetic split with the same layout and a gold.jsonl file for local development and tests. Synthetic data is not a substitute for the official training or public split.
