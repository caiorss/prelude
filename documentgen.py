#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re


def mainf(function):

    if __name__ == "__main__" :
        function()

    return function


def read_file(filename):
    return open(filename).read()

def write_file(filename, content):
    fp = open(filename, "w")
    fp.write(content)
    fp.close()




text = read_file("README.md")

re.sub(r"^# (.+)", r"<h1>\1</h1>", text, re.M)


def replacer(pattern, repl, flags=None):

    patobj = re.compile(pattern, flags=flags)
    return lambda text: patobj.sub(repl, text)

def findre(pattern, flags=None):
    patobj = re.compile(pattern, flags=flags)
    return lambda text: patobj.findall(text)





def h_formatter(level):
    repl = r"<h{level}>\1</h{level}>".format(level=level)
    pattern = r"^{}\s+(.+)".format(level * "#")

    #print(pattern)

    # lambda text: re.sub(pattern, repl, text, flags=re.M)
    return replacer(pattern, repl, re.M)


def compose_pipe(funclist):
    def _(*args):
        value = args

        for f in funclist:
            value = f(value)

        return value

    return _


h1_format = h_formatter(1)
h2_format = h_formatter(2)
h3_format = h_formatter(3)
h4_format = h_formatter(4)
h5_format = h_formatter(5)


block_code_formater = replacer(
    "```\s*\n(.*?)```",
    r"<pre>\n<code>\n\1</code>\n</pre>",
    flags=re.M+re.DOTALL
)

block_code_formater2 = replacer(
    "```(?!\n)(.+?)\s*\n(.*?)\n```",
r"""
<pre class="prettyprint">
<code  class="language-\1">
\2
</code>
</pre>
""",
    flags=re.M + re.DOTALL
)


code_generator = compose_pipe(
    [
     block_code_formater2,
     h1_format,
     h2_format,
     h3_format,
     h4_format,
     h4_format,
     block_code_formater,
    ]
)

@mainf
def make_doc():
    template = read_file("template.html")
    output = template.replace("{CONTENT}", code_generator(text))
    write_file("output.html", output)

    print("donw OK")