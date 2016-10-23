def generate_dataset(option, noise=1, noise_background=True, shuffle=False):
    """
    This function generates syntetic datasets as described in the paper
    (http://cs-people.bu.edu/panagpap/Research/Bio/bicluster_survey.pdf)
    - Figure 4.
    
    Params
        option (str): bicluster structure ('a' to 'i')
        noise (int): value of the noise in the matrix
        noise_background (bool): positions where is not a bicluster should contain noise
            if this parameter is set to True
        shuffle (bool): shuffle lines and columns of the matrix if this parameter is set
            to True
            
    Returns
        data (array_like): matrix generated
    """
    shape = (150,150)
    n,m = shape
    
    # values shouldn't be a lot far...
    centers = [20, 40, 60, 80, 100]
    
    if noise_background:
        data = np.random.rand(n, m)*100
    else:
        data = np.zeros(n*m).reshape(shape)

    if option == 'a':
        data[60:110][:,70:140] = np.random.rand(50,70)*noise + centers[0]
    elif option == 'd':
        data[0:50][:,0:70] = np.random.rand(50,70)*noise + centers[0]
        data[50:100][:,50:100] = np.random.rand(50,50)*noise + centers[2]
        data[100:150][:,80:150] = np.random.rand(50,70)*noise + centers[1]
    elif option == 'e':
        data[0:70][:,0:50] = np.random.rand(70,50)*noise + centers[3]
        data[50:100][:,50:100] = np.random.rand(50,50)*noise + centers[1]
        data[80:150][:,100:150] = np.random.rand(70,50)*noise + centers[2]
    elif option == 'f':
        data[0:50][:,0:40] = np.random.rand(50,40)*noise + centers[4]
        data[50:150][:,0:40] = np.random.rand(100,40)*noise + centers[0]
        data[110:150][:,40:95] = np.random.rand(40,55)*noise + centers[2]
        data[110:150][:,95:150] = np.random.rand(40,55)*noise + centers[1]
    elif option == 'g':
        data[0:110][:,0:40] = np.random.rand(110,40)*noise + centers[0]
        data[110:150][:,0:110] = np.random.rand(40,110)*noise + centers[2]
        data[40:150][:,110:150] = np.random.rand(110,40)*noise + centers[1]
        data[0:40][:,40:150] = np.random.rand(40,110)*noise + centers[3]
    elif option == 'h':
        data[0:90][:,0:90] = np.random.rand(90,90)*noise + centers[0]
        data[35:55][:,35:55] = (np.random.rand(20,20)*noise + centers[1]) + data[35:55][:,35:55]
        data[110:140][:,35:90] = np.random.rand(30,55)*noise + centers[4]
        data[0:140][:,110:150] = np.random.rand(140,40)*noise + centers[2]
        data[0:55][:,130:150] = (np.random.rand(55,20)*noise + centers[3]) + data[0:55][:,130:150]
    elif option == 'i':
        data[20:70][:,20:70] = np.random.rand(50,50)*noise + centers[0]
        data[20:70][:,100:150] = np.random.rand(50,50)*noise + centers[1]
        data[50:110][:,50:120] = np.random.rand(60,70)*noise + centers[2]
        data[120:150][:,20:100] = np.random.rand(30,80)*noise + centers[3]

    if shuffle:
        np.random.shuffle(data)
        np.random.shuffle(data.T)

    return data

