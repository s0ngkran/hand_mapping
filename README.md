# gen_gts
you should test sigma first. Then, gen_gts_folder()
```python
if __name__ == "__main__":
    gt_folder, dist_folder, dim1, dim2, sigma = 'testing/pkl/', 'testing/gts/', (360,360),(45,45),18
    gen_gts_folder(gt_folder, dist_folder, dim1, dim2, sigma)
```
![](https://github.com/s0ngkran/hand_mapping/blob/master/example/ex_gen_gts.png)
