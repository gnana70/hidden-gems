# Code from our YouTube videos

Companion code for our video series. Every **section** is self-contained: it has its own README with the setup,
its own scripts, its own `.env.example` for any keys, and the raw results we show on screen. Pick a folder and
follow its README; nothing in one section depends on another.

| Series | What it is | Folder |
|---|---|---|
| **Hidden Gems** | unique, under-hyped open AI models, tested hands-on on a gaming laptop, with an honest verdict | [`hidden-gems/`](hidden-gems/) |

## Layout
```
<series>/                 one folder per series, with a README that lists its episodes
  <NN>-<name>/            one folder per episode or project: README, setup, scripts, .env.example, results
_template/                copy this to start a new section
```

## Keys and tokens
Never commit them. Each section that needs one has a `.env.example`: copy it to `.env` in the same folder and fill it in.
The root `.gitignore` keeps every `.env` (and common key and credential files) out of git in all folders.

## Licence
Our code is MIT (see `LICENSE`). The models, datasets and tools a section uses keep their own licences; its README links them.
