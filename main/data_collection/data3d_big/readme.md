# data3d
時系列学習させるときようにデータ整形されたもの.  
SKIP_FRAMEごとのフレームを, TIME_SEQUENCEコ時間方向に並べたものを1つの入力とする.

## データ形式
|   name   |               shape                |
| :------: | :--------------------------------: |
| input_3d | batch_size x time_sequence x h x w |
|  label   |             batch_size             |

## 注意
3dデータはめちゃくちゃデカくなる.  
10コもの文字しかないのに, 3.3Gもある.  
データ形式をfloat16とかにして、サイズを小さくしたほうが良い。