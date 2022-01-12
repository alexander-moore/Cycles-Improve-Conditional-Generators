def return_res(model_type, directory):
	from sklearn.decomposition import PCA
	import torch
	import g_arches
	import os
	import numpy as np

	torch.cuda.empty_cache()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	def get_codes(size, hardware = device, hot = True):
	    if hot == True:
	        return one_hot_embedding(torch.randint(c_dim, size = (size, 1), device = hardware)).float()

	    else:
	        return torch.randint(c_dim, size = (size, 1), device = hardware)

	def one_hot_embedding(labels, hot = True):
	    #y = torch.eye(num_classes)
	    #return y[labels]
	    #return torch.nn.functional.one_hot(labels)[:,1:]

	    labels = torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64), num_classes = c_dim)
	    if hot == True:
	        return torch.squeeze(labels).to(device)
	    else:
	        return torch.squeeze(labels).to(device)



	v_dim = 100
	c_dim = 10
	cifar_subset_num = 2000
	#directory = 'model_sets/c_gan_cifar/'
	directory = directory
	reses = []
	#G = g_arches.rgb_32_C_G(v_dim, c_dim)
	G = model_type
	with torch.no_grad():
	    for filename in os.listdir(directory):
	        print(directory+filename)
	        print(torch.load(directory+filename))
	        a = torch.load(directory+filename)
	        G.load_state_dict(a)
	        G = G.eval()
	        G = G.to(device)
	        v = torch.randn((cifar_subset_num, v_dim, 1, 1), device = device)
	        codes = get_codes(cifar_subset_num).view(cifar_subset_num, c_dim, 1, 1)

	        outputs = G(v, codes).view(cifar_subset_num, -1).cpu()
	        pca_mod = PCA().fit(outputs)

	        cumsums = np.cumsum(pca_mod.explained_variance_ratio_)
	        res = next(x for x, val in enumerate(cumsums) 
	                                      if val > 0.95)
	        print ("The index of element just greater than 0.95 : "
	                                               + str(res))
	        reses.append(res)

	return sum(reses) / len(reses)