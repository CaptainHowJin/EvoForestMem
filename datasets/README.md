# Ever-Stream and Group-Stream

Release 1.0 provides streaming question-answering datasets for multi-party conversations. Ever-Stream contains 2,866 questions and 51,023 messages across five conversation partitions. Group-Stream contains 614 questions and 120,000 messages across four domains.

## Question identifiers

Ever-Stream uses `EVER-0001` through `EVER-2866`. Group-Stream uses `GROUP-0001` through `GROUP-0614`. Question identifiers are unique across the release. Each partition retains its chronological question order, and evaluation records use `(subset, qa_id)` as identity.

| Dataset | Partition | Questions | QA identifiers |
|---|---|---:|---|
| Ever-Stream | 004 | 560 | `EVER-0001`–`EVER-0560` |
| Ever-Stream | 005 | 560 | `EVER-0561`–`EVER-1120` |
| Ever-Stream | 010 | 575 | `EVER-1121`–`EVER-1695` |
| Ever-Stream | 011 | 595 | `EVER-1696`–`EVER-2290` |
| Ever-Stream | 016 | 576 | `EVER-2291`–`EVER-2866` |
| Group-Stream | Finance | 175 | `GROUP-0001`–`GROUP-0175` |
| Group-Stream | Technology | 176 | `GROUP-0176`–`GROUP-0351` |
| Group-Stream | Manufacturing | 146 | `GROUP-0352`–`GROUP-0497` |
| Group-Stream | Healthcare | 117 | `GROUP-0498`–`GROUP-0614` |

## Data format

Each partition contains `messages.jsonl` and `qas.jsonl`. `manifest.json` lists the partition paths, question identifiers and SHA-256 checksums. Group domains are evaluated using separate memory stores.

Ever messages use `msg_id`, `time`, `speaker`, `group` and `dialogue`. Group messages use `msg_node`, `timestamp`, `author`, `project` and `content`. The runtime maps these fields into its message representation.

Questions include `qa_id`, `question`, `answer`, `ask_time`, task categories and evidence references. Ever evidence uses `evidence_msg_ids`, and Group evidence uses `evidence_msg_nodes`. Multiple-choice questions include `options`. Where present, `id` is an alias of `qa_id`. The optional `premature_of` field links a premature question to another question in this release.

`ask_time` defines the visible message prefix. Timestamps without offsets use UTC. Question batches are derived from their `ask_time` values.

## Sources

These datasets adapt EverMemBench and GroupMemBench for timestamped streaming evaluation, with unified task categories and additional premature, stale-answer and update questions.

- EverMemBench repository, EverMind-AI/EverMemBench, revision `e10b3d52f0e4cfc5c124ad406b5d95c59c73738b`.
- EverMemBench-Dynamic data, EverMind-AI/EverMemBench-Dynamic, revision `a6b210a32248e841967b7b64a64281d2ff3f669d`.
- GroupMemBench, UCSB-NLP-Chang/GroupMemBench, revision `e2682e01ff490acfe4fac2940159dce60307dfc9`.

Dataset use follows the respective upstream terms. The repository's MIT license covers its code.
