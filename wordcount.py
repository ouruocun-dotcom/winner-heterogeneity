import re, sys

def strip_tex(s):
    s = re.sub(r'(?<!\\)%.*', '', s)                       # comments
    s = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', ' ', s, flags=re.S)
    s = re.sub(r'\\\[.*?\\\]', ' ', s, flags=re.S)
    s = re.sub(r'\$\$.*?\$\$', ' ', s, flags=re.S)
    # inline math: match $...$ only when it does not span a blank line
    s = re.sub(r'\$(?:[^$\n]|\n(?!\s*\n))*?\$', ' MATH ', s)
    s = re.sub(r'\\[a-zA-Z@]+\*?(\[[^\]]*\])?', ' ', s)    # commands
    s = re.sub(r'[{}~&\\^_]', ' ', s)
    return s

def words(s):
    return len([x for x in strip_tex(s).split() if re.search(r'[A-Za-z0-9]', x)])

t = open(sys.argv[1], encoding='utf-8').read()
figs = re.findall(r'\\begin\{figure\}(.*?)\\end\{figure\}', t, re.S)
tabs = re.findall(r'\\begin\{table\}(.*?)\\end\{table\}', t, re.S)
capf = [words(re.search(r'\\caption\{(.*)\}\s*$', f.strip(), re.S).group(1)) for f in figs]
capt = [words(re.search(r'\\caption\{(.*?)\}\s*\\begin\{ruledtabular\}', x, re.S).group(1)) for x in tabs]
ab = words(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S).group(1))
b = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', ' ', t, flags=re.S)
b = re.sub(r'\\begin\{table\}.*?\\end\{table\}', ' ', b, flags=re.S)
b = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', ' ', b, flags=re.S)
b = re.sub(r'\\begin\{acknowledgments\}.*?\\end\{acknowledgments\}', ' ', b, flags=re.S)
bw = words(b[b.index('\\maketitle'):])
neq = len(re.findall(r'\\begin\{equation\}', t))
fig_allow = len(figs) * (150/(11/4.3) + 20)
tab_allow = 26 + 13*6
tot = bw + ab + sum(capf) + sum(capt) + 16*neq + fig_allow + tab_allow
print(f"  body          {bw}")
print(f"  abstract      {ab}")
print(f"  fig captions  {sum(capf)}  {capf}")
print(f"  table caption {sum(capt)}")
print(f"  equations     {neq} -> {16*neq}")
print(f"  allowances    figures {fig_allow:.0f}, table {tab_allow}")
print(f"  TOTAL {tot:.0f} / 3750    headroom {3750-tot:.0f}")
