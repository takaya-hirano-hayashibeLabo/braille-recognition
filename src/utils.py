import numpy as np

def mask_pixels(pixels:np.ndarray,mask_size:tuple,mask_pivot:tuple,mask_val=0)->np.ndarray:
    """
    mask_pivotを左上として、2次元データに対してマスクをかける.
    チャンネル方向にはすべてマスクをする.
    
    :param pixels: もとの入力データ. [batch x chennel x width x height]
    :param mask_size: マスクのサイズ. (width:列方向, height:行方向)
    :param mask_pivot: マスクの起点. ここがマスクの左上になる. (x:列方向,y:行方向)
    :param mask_val: マスクしたときの値. 基本的に0で良いと思う.
    """

    pivot_x,pivot_y=mask_pivot
    size_x,size_y=mask_size

    mask_xx,mask_yy=np.meshgrid(
        np.arange(pivot_x,pivot_x+size_x,1),
        np.arange(pivot_y,pivot_y+size_y,1)
    )

    mask=np.zeros_like(pixels)
    mask[:,:,mask_yy,mask_xx]=1

    masked_pixels=pixels*(1-mask) + mask_val*mask #maskのないとこはそのまま. maskのあるとこはmask_valになる

    return masked_pixels


if __name__=="__main__":

    # >> test mask_pixels >>
    raw_data=np.arange(2*16).reshape(2,1,4,4)
    mask_pixels(
        pixels=raw_data,
        mask_size=(3,2),
        mask_pivot=(1,0)
    )
    # >> test mask_pixels >>
