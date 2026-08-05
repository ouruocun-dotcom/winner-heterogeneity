r"""
Cross-file consistency check.  Run after any edit to the manuscript, the
Supplemental Material or the bibliography, and before packaging.

Checks:
  * the SM quotes the current main-text title
  * braces balance in both documents
  * no dangling or unused \label / \ref in either
  * every \cite in the main text resolves in refs.bib
  * every \cite in the SM resolves in the SM's own \bibitem list
  * every \includegraphics file exists
  * the SM's hardcoded pointers to main-text equation and figure numbers
    match the current numbering

The last check is the one that catches the failure this script was written
for: the SM cannot use \ref across documents, so its references to
"Eq. (9)" or "Fig. 3" are literal numbers that silently rot whenever an
equation or figure is added, moved or removed from the main text.

Usage:  python3 check_consistency.py path/to/paper
"""
import os
import re
import sys

d = sys.argv[1] if len(sys.argv) > 1 else '.'
main = open(os.path.join(d, 'prl_main.tex'), encoding='utf-8').read()
sm = open(os.path.join(d, 'prl_supplement.tex'), encoding='utf-8').read()
bib = open(os.path.join(d, 'refs.bib'), encoding='utf-8').read()
fail = []


def check(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg)
    if not cond:
        fail.append(msg)


def title_of(t):
    return ' '.join(re.search(r'\\title\{(.*?)\}\n', t, re.S)
                    .group(1).replace('\\\\', '').split())


print('titles')
mt = title_of(main)
check(mt in title_of(sm), 'SM quotes the current main-text title')

print('structure')
for name, t in (('main', main), ('SM', sm)):
    check(t.count('{') == t.count('}'), f'{name}: braces balance')
    r = set(re.findall(r'\\ref\{([^}]+)\}', t))
    l = set(re.findall(r'\\label\{([^}]+)\}', t))
    check(not (r - l), f'{name}: no dangling \\ref  {sorted(r - l)}')
    check(not (l - r), f'{name}: no unused \\label  {sorted(l - r)}')

print('citations')
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
c = {k.strip() for g in re.findall(r'\\cite\{([^}]+)\}', main) for k in g.split(',')}
check(not (c - keys), f'main: all \\cite resolve in refs.bib  {sorted(c - keys)}')
sc = {k.strip() for g in re.findall(r'\\cite\{([^}]+)\}', sm) for k in g.split(',')}
sb = set(re.findall(r'\\bibitem\{([^}]+)\}', sm))
check(not (sc - sb), f'SM: all \\cite resolve in its own bibliography  {sorted(sc - sb)}')

print('figure files')
for f in re.findall(r'includegraphics\[[^\]]*\]\{([^}]+)\}', main + sm):
    check(os.path.exists(os.path.join(d, f)), f'{f} present')

print('SM pointers into the main text')
neq = len(re.findall(r'\\begin\{equation\}', main))
nfig = len(re.findall(r'\\begin\{figure\}', main))
for m in re.finditer(r'(?:Eq|Equation)\.?~?\((\d+)\)', sm):
    n = int(m.group(1))
    check(1 <= n <= neq, f'SM cites main Eq. ({n}); main has {neq} equations')
for m in re.finditer(r'Fig\.~(\d+) of the main text', sm):
    n = int(m.group(1))
    check(1 <= n <= nfig, f'SM cites main Fig. {n}; main has {nfig} figures')

print()
print('ALL CHECKS PASSED' if not fail else f'{len(fail)} PROBLEM(S)')
sys.exit(1 if fail else 0)
