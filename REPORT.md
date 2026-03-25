# 2.1

(a)
'\x00'

(b)
Its string representation is empty string.

(c)
It will print nothing.

# 2.2

(a)
Bytes sequence encoded with UTF-8 will be much shorter, compared to UTF-16 and UTF-32.

(b)
'你好'
Because in this function, a single byte corresponds to a single character, which is not the case, especially in non-English languages. UTF-8 could encode one character into 1 to 4 bytes.

(c)
[bytes('0x7F'), bytes('0x7F')]

# 2.5

## Problem 1

(a)
Time: 585.61 s, Memory: 33.504 GB, longest token: 'Ġaccomplishment'.
It makes sense.

(b)
The merge computation process takes the most time.

## Problem 2

(a)
Time: 256640.34 s, Memory: 164.563 GB, longest token: 'ÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤ'.
It doesn't make sense.

(b)
At lower indices, both vocabularies contains some common words. For example, 'Ġt' is 257 and 'he' is 259 in both vocabs.
But generally, tokens differ a lot, especially in higher indices.

# 2.7

(a)
TinyStories tokenizer on TinyStories samples: compression ratio 4.14.
OpenWebText tokenizer on OpenWebText samples: compression ratio 4.36.

(b)
TinyStories tokenizer on OpenWebText samples: compression ratio 3.25.
There's a significant drop.

(c)
Assume we use TinyStories tokenizer, whose throughput is 868425.42 bytes pre second.
So it would take approximately 1019 seconds (about 17 minutes).

(d)
2^15 = 32,768, which is just a little higher than the vocab size of 32,000.
