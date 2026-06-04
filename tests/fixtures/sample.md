# doc-rag legacy .doc parser fixture

This file is the source for `tests/fixtures/sample.doc`, the binary
Word-97 document used by `tests/test_parsers.py::test_parse_doc_via_antiword`.

Please convert this `.md` (or its plain-text equivalent) to legacy `.doc`
in whichever way is convenient — Microsoft Word "Save As → Word 97–2003
Document", LibreOffice Writer "File → Export as → Word 97–2003 (.doc)",
`soffice --headless --convert-to doc`, etc. The resulting file goes next
to this one as `sample.doc` and is committed to the repository.

## Why we ship a binary fixture instead of generating one in the test

The legacy `.doc` format is a closed binary container (Compound File
Binary, also known as CFB or MS-CFB) that no maintained Python library
writes from scratch. The only practical generators are Microsoft Word
and LibreOffice headless, neither of which we want as a CI dependency.
A small (≈10 kB) pre-built fixture is the cheapest path to actual test
coverage of the `antiword` parser branch on CI.

## Marker the test looks for

The parser test asserts that the marker below appears in the extracted
text. **Keep it exact** when re-generating the fixture:

```
legacy-doc-marker-pwzxa-58302
```

That string is opaque to a human but unique enough to never collide
with anything else `antiword` might emit.

## Filler — keeps the file above antiword's minimum size

`antiword` refuses to operate on very small `.doc` streams with the
message "text stream too small to handle". A few hundred bytes of real
prose are enough to satisfy it. The following paragraphs serve no other
purpose; treat them as ballast.

This document is part of the test fixtures of the doc-rag project. It is
not a real engineering specification, it is not a real standard, and it
is not intended to be read for content. Its only job is to give the
legacy .doc parser something concrete to work with so that the
end-to-end pipeline from `parse_document` through `antiword` to the
extracted-text invariant can be exercised on every CI run.

The doc-rag project parses engineering documents — standards (СТО),
state standards (ГОСТы), regulatory guidance documents (РД),
specifications, operating manuals, and explanatory notes — and exposes
them to LLM assistants via an MCP server. The pipeline supports five
input formats: PDF (with optional OCR via Tesseract), DOCX, this legacy
.doc format, Markdown, and plain text. Each format has its own parser,
its own normalisation rules, and its own corner cases. The .doc parser
shells out to `antiword` (or, as a fallback, `catdoc`) because nothing
in the Python ecosystem reliably handles the binary CFB container.

If you read this far you have read more of this file than its intended
audience, which is a parser. Thank you for your patience. The marker
above is what matters.
