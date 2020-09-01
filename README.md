## gen_gts.py
you should test_sigma() first to get sigma value. Then, put that sigma value into gen_gts_folder()
```python
if __name__ == "__main__":
    gt_file, dist_folder, dim1, dim2, sigma = 'testing/gt.torch', 'testing/gts/', (360,360),(45,45),18
    gen_gts_folder(gt_file, dist_folder, dim1, dim2, sigma)
```
## gen_gtl.py
```python
if __name__ == "__main__":
    gt_file = 'testing/gt.torch'
    savefolder = 'testing/gtl/'
    dim1 = (360,360)
    dim2 = (45,45)
    size = 10
    gen_gtl_folder(gt_file, savefolder, dim1, dim2, size)
```
![](https://github.com/s0ngkran/hand_mapping/blob/master/example/ex_gen_gt.png)
