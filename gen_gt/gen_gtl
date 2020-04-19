import torch
link = [[0,1] ,[0,3] ,[0,5] ,[0,7] ,[0,9], [1,2], [3,4], [5,6], [7,8], [9,10]]
def gt_vec(width, height, point, link=link):
    ans = torch.zeros( len(link)*2, width, height)
    x, y = np.where(ans[0]==0)
    for index, (partA, partB) in enumerate(link):
        vec = point[partB]-point[partA]
        length = np.sqrt(vec[0]**2+vec[1]**2)
        u_vec = vec/length
        u_vec_p = np.array([u_vec[1], -u_vec[0]])

        tempx = x-point[partA][0]
        tempy = y-point[partA][1]
        temp_ = []
        temp_.append(tempx)
        temp_.append(tempy)
        temp = np.stack(temp_)
  
        c1 = np.dot(u_vec,temp)
        c1 = (0<=c1) & (c1<=length)
        c2 = abs(np.dot(u_vec_p,temp)) <= 7
        condition = c1 & c2
       
        ans[ index*2] = torch.tensor(u_vec[0] * condition).reshape(width, height)  #x
        ans[ index*2+1] = torch.tensor(u_vec[1] * condition).reshape(width, height) #y
    return ans
