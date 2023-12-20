# train.py
とりあえずこれで、CNNもCSNNも学習可能.  
~~~bash
python train.py --net_type nn
python train.py --net_type snn
~~~

## テスト学習結果
学習結果としては, どっちのネットワークでもテストデータで100%達成した.

学習時のパラメータ  
|       param        |  value  |
| :----------------: | :-----: |
|     input size     | 28 x 28 |
|       batch        |   32    |
|         lr         |  0.01   |
|       epoch        |   10    |
| num_steps(snnのみ) |   24    |

学習結果  
※SNNは連続値のままNNにぶちこんでる

| net type | test accuracy |
| :------: | :-----------: |
|    nn    |     100 %     |
|   snn    |     100 %     |