# Corpus licenses

The files in `corpus/raw/` and `corpus/extra/` are normalised snapshots of public documentation pages (inline SVG diagrams, scripts and styles removed). Each document's source URL, retrieval date and hashes are recorded in `manifest.json` and `manifest.lock.json`.

## SQLite documentation (doc ids `sqlite-*`)

Source: https://www.sqlite.org/

> All of the code and documentation in SQLite has been dedicated to the public domain by the authors.
> — https://www.sqlite.org/copyright.html

## PostgreSQL documentation (doc ids `pg-*`)

Source: https://www.postgresql.org/docs/17/ (pages listed in `manifest.json`). Reproduced under the PostgreSQL License, whose notice must accompany all copies:

> PostgreSQL Database Management System
> (also known as Postgres, formerly known as Postgres95)
>
> Portions Copyright © 1996-2026, PostgreSQL Global Development Group
>
> Portions Copyright © 1994, The Regents of the University of California
>
> Permission to use, copy, modify, and distribute this software and its documentation for any purpose, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and this paragraph and the following two paragraphs appear in all copies.
>
> IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS-IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
>
> — https://www.postgresql.org/docs/17/legalnotice.html

## Documents added by users

Files uploaded through the dashboard or `python -m sqlrag ingest --files` are stored in `corpus/uploads/` (not committed). The uploader states the license or reuse terms, which are stored with every chunk. Only add documents you are allowed to reuse.
