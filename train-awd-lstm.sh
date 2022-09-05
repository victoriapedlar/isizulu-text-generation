pip install -qr requirements.txt

./utils/getdata.sh

python3 -u awd_lstm/main.py \
    --save "AWD_LSTM_5_Sept.pt" \
    --descriptive_name "AWDLSTM_Luc_Hayward" \
    --data data/autshumato/isizulu/ \
    --save_history "log_history.txt" \
    --emsize 800 \
    --nhid 1150 \
    --nlayers 3 \
    --lr 30.0 \
    --clip 0.25 \
    --epochs 750 \
    --batch_size 80 \
    --bptt 70 \
    --dropout 0.4 \
    --dropouth 0.2 \
    --dropouti 0.65 \
    --dropoute 0.1 \
    --wdrop 0.5 \
    --seed 1882 \
    --nonmono 8 \