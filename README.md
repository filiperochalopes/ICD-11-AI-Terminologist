```sh
conda create -n langgraph python=3.11 -y
conda activate langgraph
conda install -c conda-forge compilers cmake
pip install --prefer-binary -r requirements.txt
langgraph dev --host 0.0.0.0 --no-browser
langgraph run V3:app
```

```sh
# crie uma sessão chamada "lg"
tmux new -s lg

# dentro dela, rode:
conda activate langgraph
langgraph dev --host 0.0.0.0 --no-browser

# para “desanexar” sem parar o processo:
Ctrl-b d

# depois você pode voltar:
tmux attach -t lg

# e para matar:
tmux kill-session -t lg
```