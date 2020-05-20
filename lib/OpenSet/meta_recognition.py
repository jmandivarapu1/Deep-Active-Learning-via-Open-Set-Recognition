import torch
import numpy as np
import libmr
import collections
import visualization
import lib.Datasets.datasets as datasets


def sample(values,all_indices,method,budget):

    if method!='NONE':
        all_preds = values
        all_indices = all_indices
        # all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices].astype(np.int32)

        return querry_pool_indices
    else:
        return all_indices
def get_means(tensors_list):
    """
    Calculate the mean of a list of tensors for each tensor in the list. In our case the list typically contains
    a tensor for each class, such as the per class z values.

    Parameters:
        tensors_list (list): List of Tensors

    Returns:
        list: List of Tensors containing mean vectors
    """

    means = []
    for i in range(len(tensors_list)):
        if isinstance(tensors_list[i], torch.Tensor):
            means.append(torch.mean(tensors_list[i], dim=0))
        else:
            means.append([])

    return means


def calc_distances_to_means(means, tensors, distance_function='cosine'):
    """
    Function to calculate distances between tensors, in our case the mean zs per class and z for each input.
    Wrapper around torch.nn.functonal distances with specification of which distance function to choose.

    Parameters:
        means (list): List of length corresponding to number of classes containing torch tensors (typically mean zs).
        tensors (list): List of length corresponding to number of classes containing tensors (typically zs).
        distance_function (str): Specification of distance function. Choice of cosine|euclidean|mix.

    Returns:
        list: List of length corresponding to number of classes containing tensors with distance values
    """

    def distance_func(a, b, w_eucl, w_cos):
        """
        Weighted distance function consisting of cosine and euclidean components.

        Parameters:
            a (torch.Tensor): First tensor.
            b (torch.Tensor): Second tensor.
            w_eucl (float): Weight for the euclidean distance term.
            w_cos (float): Weight for the cosine similarity term.
        """
        d = w_cos * (1 - torch.nn.functional.cosine_similarity(a.view(1, -1), b)) + \
            w_eucl * torch.nn.functional.pairwise_distance(a.view(1, -1), b, p=2)
        return d

    distances = []

    # weight values for individual distance components
    w_eucl = 0.0
    w_cos = 0.0
    if distance_function == 'euclidean':
        w_eucl = 1.0
    elif distance_function == 'cosine':
        w_cos = 1.0
    elif distance_function == 'mix':
        w_eucl = 0.5
        w_cos = 0.5
    else:
        raise ValueError("distance function not implemented")

    # loop through each class in means and calculate the distances with the respective tensor.
    for i in range(len(means)):
        # check for tensor type, e.g. list could be empty
        if isinstance(tensors[i], torch.Tensor) and isinstance(means[i], torch.Tensor):
            distances.append(distance_func(means[i], tensors[i], w_eucl, w_cos))
        else:
            distances.append([])

    return distances


def fit_weibull_models(distribution_values, tailsizes, num_max_fits=50):
    """
    Function to fit weibull models on distribution values per class. The distribution values in our case are the
    distances of an inputs approximate posterior value to the per class mean latent z, i.e. The Weibull model fits
    regions of high density and gives credible intervals.
    The tailsize specifies how many outliers are expected in the dataset for which the model has been trained.
    We use libmr https://github.com/Vastlab/libMR (installable through e.g. pip) for the Weibull model fitting.

    Parameters:
        distribution_values (list): Values on which the fit is conducted. In our case latent space distances.
        tailsizes (list): List of integers, specifying tailsizes per class. For a balanced dataset typically the same.
        num_max_fits (int): Number of attempts to fit the Weibull models before timing out and returning unsuccessfully.

    Returns:
        list: List of Weibull models with their respective parameters (stored in libmr class instances).
    """

    weibull_models = []

    # loop through the list containing distance values per class
    for i in range(len(distribution_values)):
        # for each class set the initial success to False and number of attempts to 0
        is_valid = False
        count = 0

        # If the list contains distance values conduct a fit. If it is empty, e.g. because there is not a single
        # prediction for the corresponding class, continue with the next class. Note that the latter isn't expected for
        # a model that has been trained for even just a short while.
        if isinstance(distribution_values[i], torch.Tensor):
            distribution_values[i] = distribution_values[i].cpu().numpy()
            # weibull model per class
            weibull_models.append(libmr.MR())
            # attempt num_max_fits many fits before aborting
            while is_valid is False and count < num_max_fits:
                # conduct the fit with libmr
                weibull_models[i].fit_high(distribution_values[i], tailsizes[i])
                is_valid = weibull_models[i].is_valid
                count += 1
            if not is_valid:
                print("Weibull fit for class " + str(i) + " not successful after " + str(num_max_fits) + " attempts")
                return weibull_models, False
        else:
            weibull_models.append([])

    return weibull_models, True


def calc_outlier_probs(weibull_models, distances):
    """
    Calculates statistical outlier probability using the weibull models' CDF.

    Note that we have coded this function to loop over each class because we have previously categorized the distances
    into their respective classes already.

    Parameters:
        weibull_models (list): List of libmr class instances containing the Weibull model parameters and functions.
        distances (list): List of per class torch tensors or numpy arrays with latent space distance values.

    Returns:
        list: List of length corresponding to number of classes with outlier probabilities for each respective input.
    """

    outlier_probs = []
    # loop through all classes, i.e. all available weibull models as there is one weibull model per class.
    for i in range(len(weibull_models)):
        # optionally convert the type of the distance vectors
        if isinstance(distances[i], torch.Tensor):
            distances[i] = distances[i].cpu().numpy().astype(np.double)
        elif isinstance(distances[i], list):
            # empty list
            outlier_probs.append([])
            continue
        else:
            distances[i] = distances[i].astype(np.double)

        # use the Weibull models' CDF to evaluate statistical outlier rejection probabilities.
        outlier_probs.append(weibull_models[i].w_score_vector(distances[i]))

    return outlier_probs


def calc_openset_classification(data_outlier_probs, num_classes, num_outlier_threshs=50):
    """
    Calculates the percentage of dataset outliers given a set of outlier probabilities over a range of rejection priors.

    Parameters:
         data_outlier_probs (list): List of outlier probabilities for an entire dataset, categorized by class.
         num_classes (int): Number of classes.
         num_outlier_threshs (int): Number of outlier rejection priors (evenly spread over the interval (0,1)).

    Returns:
        dict: Dictionary containing outlier percentages and corresponding rejection prior values.
    """

    dataset_outliers = []
    threshs = []

    # loop through each rejection prior value and evaluate the percentage of the dataset being considered as
    # statistical outliers, i.e. each data point's outlier probability > rejection prior.
    for i in range(num_outlier_threshs - 1):
        outlier_threshold = (i + 1) * (1.0 / num_outlier_threshs)
        threshs.append(outlier_threshold)

        dataset_outliers.append(0)
        total_dataset = 0

        for j in range(num_classes):
            total_dataset += len(data_outlier_probs[j])

            for k in range(len(data_outlier_probs[j])):
                if data_outlier_probs[j][k] > outlier_threshold:
                    dataset_outliers[i] += 1

        dataset_outliers[i] = dataset_outliers[i] / float(total_dataset)

    return {"thresholds": threshs, "outlier_percentage": dataset_outliers}


def calc_entropy_classification(dataset_entropies, max_thresh_value,args, num_outlier_threshs=50):
    """
    Calculates the percentage of dataset outliers given a set of entropies over a range of rejection priors.
    Parameters:
         dataset_entropies (list): List of entropies for the entire dataset (each instance)
         num_outlier_threshs (int): Number of outlier rejection priors (evenly spread over the interval (0,1)).
    Returns:
        dict: Dictionary containing outlier percentages and corresponding rejection prior values.
    """

    dataset_outliers = []
    threshs = []

    total_dataset = float(len(dataset_entropies))

    # loop through each rejection prior value and evaluate the percentage of the dataset being considered as
    # statistical outliers, i.e. each data point's outlier probability > rejection prior.
    for i in range(num_outlier_threshs - 1):
        outlier_threshold = (i + 1) * (max_thresh_value / num_outlier_threshs)
        threshs.append(outlier_threshold)

        dataset_outliers.append(0)

        for k in range(len(dataset_entropies)):
            if dataset_entropies[k] > outlier_threshold:
                dataset_outliers[i] += 1

        dataset_outliers[i] = dataset_outliers[i] / total_dataset

    return {"entropy_thresholds": threshs, "entropy_outlier_percentage": dataset_outliers}


def eval_var_openset_dataset(model, data_loader, num_classes, device,args, latent_var_samples=1, model_var_samples=1):
    """
    Evaluates an entire dataset with the variational or joint model and stores z values, latent mus and sigmas and
    output predictions such that they can later be used for statistical outlier evaluation with the fitted Weibull
    models. This is merely for convenience to keep the rest of the code API the same. Note that the Weibull model's
    prediction of whether a sample from an unknown dataset is a statistical outlier or not can be done on an instance
    level. Similar to the eval_dataset function but without splitting of correct vs. false predictions as the dataset
    is unknown in the open-set scenario.
    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        latent_var_samples (int): Number of latent space variational samples.
        model_var_samples (int): Number of variational samples of the entire model, e.g. MC dropout.
    Returns:
        dict: Dictionary of results and latent values.
    """

    # switch to evaluation mode unless MC dropout is active
    if model_var_samples > 1:
        model.train()
    else:
        model.eval()

    out_mus = []
    out_sigmas = []
    encoded_mus = []
    encoded_sigmas = []
    zs = []
    all_preds = []
    all_indices = []

    out_entropy = []
    collect_indexes_per_class=[]

    for i in range(num_classes):
        out_mus.append([])
        out_sigmas.append([])
        encoded_mus.append([])
        encoded_sigmas.append([])
        zs.append([])
        collect_indexes_per_class.append([])

    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for inputs, classes,indexes in data_loader:
            inputs, classes = inputs.to(device), classes.to(device)

            out_samples = torch.zeros(model_var_samples, latent_var_samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(model_var_samples, latent_var_samples,
                                    inputs.size(0), model.latent_dim).to(device)

            # sampling the model, then z and classifying
            for k in range(model_var_samples):
                encoded_mu, encoded_std,_ = model.encode(inputs)

                for i in range(latent_var_samples):
                    z = model.reparameterize(encoded_mu, encoded_std)
                    z_samples[k][i] = z

                    cl = model.classifier(z)
                    out = torch.nn.functional.softmax(cl, dim=1)
                    out_samples[k][i] = out

            # calculate the mean and std. Only removes a dummy dimension if number of variational samples is set to one.
            out_mean = torch.mean(torch.mean(out_samples, dim=0), dim=0)
            # preds = out_mean.cpu().data
            all_preds.extend(out_mean.cpu().data)
            all_indices.extend(indexes)
            if model_var_samples > 1:
                out_std = torch.std(torch.mean(out_samples, dim=0), dim=0)
            else:
                out_std = torch.squeeze(torch.std(out_samples, dim=1))
            zs_mean = torch.mean(torch.mean(z_samples, dim=0), dim=0)
            
            # calculate entropy for the means of samples: - sum pc*log(pc)
            eps = 1e-10
            out_entropy.append(- torch.sum(out_mean*torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

            # In contrast to the eval_dataset function, there is no split into correct or false values as the dataset
            # is unknown.
            for i in range(inputs.size(0)):
                idx = torch.argmax(out_mean[i]).item()
                out_mus[idx].append(out_mean[i][idx].item())
                out_sigmas[idx].append(out_std[i][idx].item())
                encoded_mus[idx].append(encoded_mu[i].data)
                encoded_sigmas[idx].append(encoded_std[i].data)
                zs[idx].append(zs_mean[i].data)
                collect_indexes_per_class[idx].append(indexes[i].item())

    # stack latent activations into a tensor
    for i in range(len(encoded_mus)):
        if len(encoded_mus[i]) > 0:
            encoded_mus[i] = torch.stack(encoded_mus[i], dim=0)
            encoded_sigmas[i] = torch.stack(encoded_sigmas[i], dim=0)
            zs[i] = torch.stack(zs[i], dim=0)

    out_entropy = np.concatenate(out_entropy).ravel().tolist()

    d={"encoded_mus": encoded_mus, "encoded_sigmas": encoded_sigmas,
            "out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs,
            "out_entropy": out_entropy,'all_preds':all_preds,    "all_indices":all_indices}

    # with open('outfile', 'wb') as fp:pickle.dump(d, fp)

    all_preds=[]
    for i in range(0,len(d['all_preds'])):
        idx=torch.argmax(d['all_preds'][i]).item()
        all_preds.append(d['all_preds'][i][idx])#.data.item())
    all_preds
    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    # need to multiply by -1 to be able to use torch.topk 
    all_preds *= -1

    # select the points which the discriminator things are the most likely to be unlabeled
    _, querry_indices = torch.topk(all_preds, int(args.budget))
    querry_pool_indices = np.asarray(all_indices)[querry_indices]

    # Return a dictionary of stored values.
    # with open('outfile', 'wb') as fp:pickle.dump(d, fp)
    return {"encoded_mus": encoded_mus, "encoded_sigmas": encoded_sigmas,
            "out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs,
            "out_entropy": out_entropy,'all_preds':all_preds,
            "all_indices":all_indices,
            'querry_pool_indices':querry_pool_indices,
            'collect_indexes_per_class':collect_indexes_per_class}

def Weibull_Sampler(model,train_loader,test_dataloader,val_loader,unlabeled_dataloader,val_dataloader_set1,val_dataloader_set2,eval_var_dataset,args,save_path):

    dataset_eval_dict_train = eval_var_dataset(model, train_loader,args.num_classes, args.device,latent_var_samples=args.var_samples, model_var_samples=args.model_samples)
    print("Training accuracy: ", dataset_eval_dict_train["accuracy"])#"accuracy"])
    #Start Preparing for the sampling
    # Get the mean of z for correctly classified data inputs
    mean_zs = get_means(dataset_eval_dict_train["zs_correct"])
    # visualize the mean z vectors
    #mean_zs_tensor = torch.stack(mean_zs, dim=0)
    # visualize_means(mean_zs_tensor, num_classes, args.dataset, save_path, "z")
    # calculate each correctly classified example's distance to the mean z
    distances_to_z_means_correct_train = calc_distances_to_means(mean_zs, dataset_eval_dict_train["zs_correct"],args.distance_function)

    #Weibull fitting
    # set tailsize according to command line parameters (according to percentage of dataset size)
    tailsize = int(len(train_loader)*128 * args.openset_weibull_tailsize / args.num_classes)
    print("Fitting Weibull models with tailsize: " + str(tailsize),len(train_loader))
    tailsizes = []#[tailsize] * args.num_classes
    for i in range(0,len(dataset_eval_dict_train["zs_correct"])):
        tailsizes.append(int(args.outlier_percentage*len(dataset_eval_dict_train["zs_correct"][i])))
    weibull_models, valid_weibull = fit_weibull_models(distances_to_z_means_correct_train, tailsizes)
    # ------------------------------------------------------------------------------------------
    # Fitting on train dataset complete. Determine rejection thresholds/priors on the created split set
    # ------------------------------------------------------------------------------------------
    print("Evaluating original threshold split dataset: " + args.dataset + ". This may take a while...")
    threshset_eval_dict = eval_var_dataset(model, val_dataloader_set1, args.num_classes, args.device,latent_var_samples=args.var_samples, model_var_samples=args.model_samples)

    # Again calculate distances to mean z
    print("Split set accuracy: ", threshset_eval_dict["accuracy"])
    distances_to_z_means_threshset = calc_distances_to_means(mean_zs, threshset_eval_dict["zs_correct"],args.distance_function)
    # get Weibull outlier probabilities for thresh set
    outlier_probs_threshset = calc_outlier_probs(weibull_models, distances_to_z_means_threshset)
    threshset_classification = calc_openset_classification(outlier_probs_threshset, args.num_classes,num_outlier_threshs=100)
    #print("threshset_classification is",threshset_classification)
    # also check outlier detection based on entropy
    max_entropy = np.max(threshset_eval_dict["out_entropy"])
    print("Max entopy is",max_entropy)
    threshset_entropy_classification = calc_entropy_classification(threshset_eval_dict["out_entropy"],max_entropy,args,num_outlier_threshs=100)
    #print("calc_entropy_classification",threshset_entropy_classification)
    # determine rejection priors based on 5% of the split data considered as inlying
    if (np.array(threshset_classification["outlier_percentage"]) <= 0.05).any() == True:
        EVT_prior_index = np.argwhere(np.array(threshset_classification["outlier_percentage"])<= 0.05)[0][0]
        EVT_prior = threshset_classification["thresholds"][EVT_prior_index]
    else:
        EVT_prior = 0.5
        EVT_prior_index = 50

    if (np.array(threshset_entropy_classification["entropy_outlier_percentage"]) <= 0.05).any() == True:
        entropy_threshold_index = np.argwhere(np.array(threshset_entropy_classification["entropy_outlier_percentage"])
                                            <= 0.05)[0][0]
        entropy_threshold = threshset_entropy_classification["entropy_thresholds"][entropy_threshold_index]
    else:
        # this should never actually happen
        entropy_threshold = np.median(threshset_entropy_classification["entropy_thresholds"])
        entropy_threshold_index = 50
    
    
    # ------------------------------------------------------------------------------------------
    # We evaluate the validation set to later evaluate trained dataset's statistical inlier/outlier estimates.
    print("Evaluating original validation dataset: " + args.dataset + ". This may take a while...")
    dataset_eval_dict = eval_var_dataset(model, val_dataloader_set2, args.num_classes, args.device,latent_var_samples=args.var_samples, model_var_samples=args.model_samples)

    # Again calculate distances to mean z
    print("Validation accuracy: ", dataset_eval_dict["accuracy"])
    distances_to_z_means_correct = calc_distances_to_means(mean_zs, dataset_eval_dict["zs_correct"],args.distance_function)

    # Evaluate outlier probability of trained dataset's validation set
    outlier_probs_correct = calc_outlier_probs(weibull_models, distances_to_z_means_correct)

    dataset_classification_correct = calc_openset_classification(outlier_probs_correct, args.num_classes,num_outlier_threshs=100)
    dataset_entropy_classification_correct = calc_entropy_classification(dataset_eval_dict["out_entropy"],max_entropy,args,num_outlier_threshs=100)

    print(args.dataset + '(trained) EVT outlier percentage: ' +str(dataset_classification_correct["outlier_percentage"][EVT_prior_index]))
    print(args.dataset + '(trained) entropy outlier percentage: ' +str(dataset_entropy_classification_correct["entropy_outlier_percentage"][entropy_threshold_index]))
    
    ##########################################################################################################################################
    #                           START ON THE RESt OF TRAINING SET
    ##########################################################################################################################################
    od=0
    openset_dataset='cifar10'
    openset_datasets_names = args.openset_datasets.strip().split(',')
    openset_sampler_methods=args.samplerMethod.strip().split(',')
    samplerMethod=args.sampler
    print("The Sampling Method is",samplerMethod)
    openset_datasets = []
    # Repeat process for open set recognition on unseen datasets (
    openset_dataset_eval_dicts = collections.OrderedDict()
    openset_outlier_probs_dict = collections.OrderedDict()
    openset_classification_dict = collections.OrderedDict()
    openset_entropy_classification_dict = collections.OrderedDict()
    print("Evaluating on rest of the tain set This may take a while...",len(test_dataloader)*128)
    openset_dataset_eval_dict = eval_var_openset_dataset(model,unlabeled_dataloader, args.num_classes,
                                                        args.device,args, latent_var_samples=args.var_samples,
                                                        model_var_samples=args.model_samples)
    


    #with open('openset_dataset_eval_dict', 'wb') as fp:pickle.dump(openset_dataset_eval_dict, fp)
    #with open('mean_zs', 'wb') as fp:pickle.dump(mean_zs, fp)
    # with open('weibull_models', 'wb') as fp:pickle.dump(weibull_models, fp)
    #with open('openset_distances_to_z_means', 'wb') as fp:pickle.dump(openset_distances_to_z_means, fp)
    # sys.exit()
    openset_distances_to_z_means = calc_distances_to_means(mean_zs, openset_dataset_eval_dict["zs"],'cosine')#args.distance_function)

    openset_outlier_probs = calc_outlier_probs(weibull_models, openset_distances_to_z_means)

    # getting outlier classification accuracies across the entire datasets
    openset_classification = calc_openset_classification(openset_outlier_probs, args.num_classes,num_outlier_threshs=100)

    openset_entropy_classification = calc_entropy_classification(openset_dataset_eval_dict["out_entropy"],max_entropy, args,num_outlier_threshs=100)

    if samplerMethod=='classifierProbability':
        topk=sample([],openset_dataset_eval_dict['querry_pool_indices'],'NONE',args.budget)
    elif  samplerMethod=='LatentMeanDistance':
        Rvalues=[]
        Rindexes=[]
        for i in range(0,args.num_classes):#openset_outlier_probs:
            Rvalues.append(torch.Tensor(openset_distances_to_z_means[i]))
            Rindexes.append(torch.Tensor(openset_dataset_eval_dict['collect_indexes_per_class'][i]))
        if  args.samplePerClass:
            print("came to sample per class")
            perclassSample=[]
            for i in range(0,args.num_classes):
                topk=sample(Rvalues[i],Rindexes[i],samplerMethod,int(args.budget/args.num_classes))
                perclassSample.append(torch.Tensor(topk))

            topk=torch.cat(perclassSample,dim=0)
        else:
            topk=sample(torch.cat(Rvalues, dim=0),torch.cat(Rindexes, dim=0),samplerMethod,args.budget)
            
    elif  samplerMethod=='WiebullOutlierProbs':
        print("came to WiebullOutlierProbs sampling")
        Rvalues=[]
        Rindexes=[]
        for i in range(0,args.num_classes):#openset_outlier_probs:
            Rvalues.append(torch.Tensor(openset_outlier_probs[i]))
            Rindexes.append(torch.Tensor(openset_dataset_eval_dict['collect_indexes_per_class'][i]))
        if  args.samplePerClass:
            perclassSample=[]
            for i in range(0,args.num_classes):
                topk=sample(Rvalues[i],Rindexes[i],samplerMethod,int(args.budget/args.num_classes))
                perclassSample.append(torch.Tensor(topk))
            topk=torch.cat(perclassSample,dim=0)
        else:
            topk=sample(torch.cat(Rvalues, dim=0),torch.cat(Rindexes, dim=0),samplerMethod,args.budget)
            
    # elif samplerMethod=='Entropy':
    #     topk=sample(openset_dataset_eval_dict,openset_classification,samplerMethod)
    # elif samplerMethod=='openSet':
    #     topk=sample(openset_dataset_eval_dict,openset_entropy_classification,samplerMethod)

    # dictionary of dictionaries: per datasetname one dictionary with respective values
    openset_dataset_eval_dicts[openset_datasets_names[od]] = openset_dataset_eval_dict
    openset_outlier_probs_dict[openset_datasets_names[od]] = openset_outlier_probs
    openset_classification_dict[openset_datasets_names[od]] = openset_classification
    openset_entropy_classification_dict[openset_datasets_names[od]] = openset_entropy_classification

    # print outlier rejection values for all unseen unknown datasets
    for other_data_name, other_data_dict in openset_classification_dict.items():
        print(other_data_name + ' EVT outlier percentage: ' +
            str(other_data_dict["outlier_percentage"][entropy_threshold_index]))

    for other_data_name, other_data_dict in openset_entropy_classification_dict.items():
        print(other_data_name + ' entropy outlier percentage: ' +
            str(other_data_dict["entropy_outlier_percentage"][entropy_threshold_index]))
        

    # joint prediction uncertainty plot for all datasets
    if (args.train_var and args.var_samples > 1) or args.model_samples > 1:
        visualization.visualize_classification_uncertainty(dataset_eval_dict["out_mus_correct"],
                                            dataset_eval_dict["out_sigmas_correct"],
                                            openset_dataset_eval_dicts,
                                            "out_mus", "out_sigmas",
                                            args.dataset + ' (trained)',
                                            args.var_samples, save_path)

    # visualize the outlier probabilities
    visualization.visualize_weibull_outlier_probabilities(outlier_probs_correct, openset_outlier_probs_dict,
                                            args.dataset + ' (trained)', save_path, tailsize)

    visualization.visualize_classification_scores(dataset_eval_dict["out_mus_correct"], openset_dataset_eval_dicts, 'out_mus',
                                    args.dataset + ' (trained)', save_path)

    visualization.visualize_entropy_histogram(dataset_eval_dict["out_entropy"], openset_dataset_eval_dicts,
                                dataset_entropy_classification_correct["entropy_thresholds"][-1], "out_entropy",
                                args.dataset + ' (trained)', save_path)

    # joint plot for outlier detection accuracy for seen and both unseen datasets
    visualization.visualize_openset_classification(dataset_classification_correct["outlier_percentage"],
                                    openset_classification_dict, "outlier_percentage",
                                    args.dataset + ' (trained)',
                                    dataset_classification_correct["thresholds"], save_path, tailsize)

    visualization.visualize_entropy_classification(dataset_entropy_classification_correct["entropy_outlier_percentage"],
                                    openset_entropy_classification_dict, "entropy_outlier_percentage",
                                    args.dataset + ' (trained)',
                                    dataset_entropy_classification_correct["entropy_thresholds"], save_path)

    return topk


def Weibull_Sampler_all_datasets(model,train_loader,test_dataloader,val_loader,unlabeled_dataloader,val_dataloader_set1,val_dataloader_set2,eval_var_dataset,args,save_path):

    dataset_eval_dict_train = eval_var_dataset(model, train_loader,args.num_classes, args.device,latent_var_samples=args.var_samples, model_var_samples=args.model_samples)
    print("Training accuracy: ", dataset_eval_dict_train["accuracy"])#"accuracy"])
    #Start Preparing for the sampling
    # Get the mean of z for correctly classified data inputs
    mean_zs = get_means(dataset_eval_dict_train["zs_correct"])
    # visualize the mean z vectors
    #mean_zs_tensor = torch.stack(mean_zs, dim=0)
    # visualize_means(mean_zs_tensor, num_classes, args.dataset, save_path, "z")
    # calculate each correctly classified example's distance to the mean z
    distances_to_z_means_correct_train = calc_distances_to_means(mean_zs, dataset_eval_dict_train["zs_correct"],args.distance_function)

    #Weibull fitting
    # set tailsize according to command line parameters (according to percentage of dataset size)
    tailsize = int(len(train_loader)*128 * args.openset_weibull_tailsize / args.num_classes)
    print("Fitting Weibull models with tailsize: " + str(tailsize),len(train_loader))
    tailsizes = [tailsize] * args.num_classes
    weibull_models, valid_weibull = fit_weibull_models(distances_to_z_means_correct_train, tailsizes)
    # ------------------------------------------------------------------------------------------
    # Fitting on train dataset complete. Determine rejection thresholds/priors on the created split set
    # ------------------------------------------------------------------------------------------
    print("Evaluating original threshold split dataset: " + args.dataset + ". This may take a while...")
    threshset_eval_dict = eval_var_dataset(model, val_dataloader_set1, args.num_classes, args.device,latent_var_samples=args.var_samples, model_var_samples=args.model_samples)

    # Again calculate distances to mean z
    print("Split set accuracy: ", threshset_eval_dict["accuracy"])
    distances_to_z_means_threshset = calc_distances_to_means(mean_zs, threshset_eval_dict["zs_correct"],args.distance_function)
    # get Weibull outlier probabilities for thresh set
    outlier_probs_threshset = calc_outlier_probs(weibull_models, distances_to_z_means_threshset)
    threshset_classification = calc_openset_classification(outlier_probs_threshset, args.num_classes,num_outlier_threshs=100)
    #print("threshset_classification is",threshset_classification)
    # also check outlier detection based on entropy
    max_entropy = np.max(threshset_eval_dict["out_entropy"])
    print("Max entopy is",max_entropy)
    threshset_entropy_classification = calc_entropy_classification(threshset_eval_dict["out_entropy"],max_entropy,args,num_outlier_threshs=100)
    #print("calc_entropy_classification",threshset_entropy_classification)
    # determine rejection priors based on 5% of the split data considered as inlying
    if (np.array(threshset_classification["outlier_percentage"]) <= 0.05).any() == True:
        EVT_prior_index = np.argwhere(np.array(threshset_classification["outlier_percentage"])<= 0.05)[0][0]
        EVT_prior = threshset_classification["thresholds"][EVT_prior_index]
    else:
        EVT_prior = 0.5
        EVT_prior_index = 50

    if (np.array(threshset_entropy_classification["entropy_outlier_percentage"]) <= 0.05).any() == True:
        entropy_threshold_index = np.argwhere(np.array(threshset_entropy_classification["entropy_outlier_percentage"])
                                            <= 0.05)[0][0]
        entropy_threshold = threshset_entropy_classification["entropy_thresholds"][entropy_threshold_index]
    else:
        # this should never actually happen
        entropy_threshold = np.median(threshset_entropy_classification["entropy_thresholds"])
        entropy_threshold_index = 50
    
    
    # ------------------------------------------------------------------------------------------
    # We evaluate the validation set to later evaluate trained dataset's statistical inlier/outlier estimates.
    print("Evaluating original validation dataset: " + args.dataset + ". This may take a while...")
    dataset_eval_dict = eval_var_dataset(model, val_dataloader_set2, args.num_classes, args.device,latent_var_samples=args.var_samples, model_var_samples=args.model_samples)

    # Again calculate distances to mean z
    print("Validation accuracy: ", dataset_eval_dict["accuracy"])
    distances_to_z_means_correct = calc_distances_to_means(mean_zs, dataset_eval_dict["zs_correct"],args.distance_function)

    # Evaluate outlier probability of trained dataset's validation set
    outlier_probs_correct = calc_outlier_probs(weibull_models, distances_to_z_means_correct)

    dataset_classification_correct = calc_openset_classification(outlier_probs_correct, args.num_classes,num_outlier_threshs=100)
    dataset_entropy_classification_correct = calc_entropy_classification(dataset_eval_dict["out_entropy"],max_entropy,args,num_outlier_threshs=100)

    print(args.dataset + '(trained) EVT outlier percentage: ' +str(dataset_classification_correct["outlier_percentage"][EVT_prior_index]))
    print(args.dataset + '(trained) entropy outlier percentage: ' +str(dataset_entropy_classification_correct["entropy_outlier_percentage"][entropy_threshold_index]))
    
    ##########################################################################################################################################
    #                           START ON THE RESt OF TRAINING SET
    ##########################################################################################################################################
    od=0
    openset_dataset='cifar10'
    openset_datasets_names = args.openset_datasets.strip().split(',')
    openset_sampler_methods=args.samplerMethod.strip().split(',')
    samplerMethod=args.sampler
    print("The Sampling Method is",samplerMethod)
    openset_datasets = []
    for openset_dataset in openset_datasets_names:
        openset_data_init_method = getattr(datasets, openset_dataset)
        openset_datasets.append(openset_data_init_method(torch.cuda.is_available(), args))
    # Repeat process for open set recognition on unseen datasets (
    openset_dataset_eval_dicts = collections.OrderedDict()
    openset_outlier_probs_dict = collections.OrderedDict()
    openset_classification_dict = collections.OrderedDict()
    openset_entropy_classification_dict = collections.OrderedDict()
    topk=[]
    for od, openset_dataset in enumerate(openset_datasets):
        if openset_datasets_names[od]=='cifar10':
            num_classes=10
        elif openset_datasets_names[od]=='CIFAR100':
            num_classes=100
        print("Evaluating openset dataset: " + openset_datasets_names[od] + ". This may take a while...",openset_dataset)
        print("Evaluating on rest of the tain set This may take a while...",len(test_dataloader)*128)
        print("Evaluating openset dataset: " + openset_datasets_names[od] + ". This may take a while...")
        openset_dataset_eval_dict = eval_var_openset_all_datasets(model, openset_dataset.val_loader, num_classes,
                                                         args.device,args, latent_var_samples=args.var_samples,
                                                         model_var_samples=args.model_samples)

        openset_distances_to_z_means = calc_distances_to_means(mean_zs, openset_dataset_eval_dict["zs"],
                                                               args.distance_function)

        openset_outlier_probs = calc_outlier_probs(weibull_models, openset_distances_to_z_means)

        # getting outlier classification accuracies across the entire datasets
        openset_classification = calc_openset_classification(openset_outlier_probs, num_classes,
                                                             num_outlier_threshs=100)

        openset_entropy_classification = calc_entropy_classification(openset_dataset_eval_dict["out_entropy"],
                                                                     max_entropy,args, num_outlier_threshs=100)

        # dictionary of dictionaries: per datasetname one dictionary with respective values
        openset_dataset_eval_dicts[openset_datasets_names[od]] = openset_dataset_eval_dict
        openset_outlier_probs_dict[openset_datasets_names[od]] = openset_outlier_probs
        openset_classification_dict[openset_datasets_names[od]] = openset_classification
        openset_entropy_classification_dict[openset_datasets_names[od]] = openset_entropy_classification

    # print outlier rejection values for all unseen unknown datasets
    for other_data_name, other_data_dict in openset_classification_dict.items():
        print(other_data_name + ' EVT outlier percentage: ' +
              str(other_data_dict["outlier_percentage"][entropy_threshold_index]))

    for other_data_name, other_data_dict in openset_entropy_classification_dict.items():
        print(other_data_name + ' entropy outlier percentage: ' +
              str(other_data_dict["entropy_outlier_percentage"][entropy_threshold_index]))

    # joint prediction uncertainty plot for all datasets
    if (args.train_var and args.var_samples > 1) or args.model_samples > 1:
        visualize_classification_uncertainty(dataset_eval_dict["out_mus_correct"],
                                             dataset_eval_dict["out_sigmas_correct"],
                                             openset_dataset_eval_dicts,
                                             "out_mus", "out_sigmas",
                                             args.dataset + ' (trained)',
                                             args.var_samples, save_path)

    # visualize the outlier probabilities
    visualization.visualize_weibull_outlier_probabilities(outlier_probs_correct, openset_outlier_probs_dict,
                                            args.dataset + ' (trained)', save_path, tailsize)

    visualization.visualize_classification_scores(dataset_eval_dict["out_mus_correct"], openset_dataset_eval_dicts, 'out_mus',
                                    args.dataset + ' (trained)', save_path)

    visualization.visualize_entropy_histogram(dataset_eval_dict["out_entropy"], openset_dataset_eval_dicts,
                                dataset_entropy_classification_correct["entropy_thresholds"][-1], "out_entropy",
                                args.dataset + ' (trained)', save_path)

    # joint plot for outlier detection accuracy for seen and both unseen datasets
    visualization.visualize_openset_classification(dataset_classification_correct["outlier_percentage"],
                                     openset_classification_dict, "outlier_percentage",
                                     args.dataset + ' (trained)',
                                     dataset_classification_correct["thresholds"], save_path, tailsize)

    visualization.visualize_entropy_classification(dataset_entropy_classification_correct["entropy_outlier_percentage"],
                                     openset_entropy_classification_dict, "entropy_outlier_percentage",
                                     args.dataset + ' (trained)',
                                     dataset_entropy_classification_correct["entropy_thresholds"], save_path)

    return topk



def eval_var_openset_all_datasets(model, data_loader, num_classes, device,args, latent_var_samples=1, model_var_samples=1):
    """
    Evaluates an entire dataset with the variational or joint model and stores z values, latent mus and sigmas and
    output predictions such that they can later be used for statistical outlier evaluation with the fitted Weibull
    models. This is merely for convenience to keep the rest of the code API the same. Note that the Weibull model's
    prediction of whether a sample from an unknown dataset is a statistical outlier or not can be done on an instance
    level. Similar to the eval_dataset function but without splitting of correct vs. false predictions as the dataset
    is unknown in the open-set scenario.
    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        latent_var_samples (int): Number of latent space variational samples.
        model_var_samples (int): Number of variational samples of the entire model, e.g. MC dropout.
    Returns:
        dict: Dictionary of results and latent values.
    """

    # switch to evaluation mode unless MC dropout is active
    if model_var_samples > 1:
        model.train()
    else:
        model.eval()

    out_mus = []
    out_sigmas = []
    encoded_mus = []
    encoded_sigmas = []
    zs = []
    all_preds = []
    all_indices = []

    out_entropy = []
    collect_indexes_per_class=[]

    for i in range(num_classes):
        out_mus.append([])
        out_sigmas.append([])
        encoded_mus.append([])
        encoded_sigmas.append([])
        zs.append([])
        collect_indexes_per_class.append([])

    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for inputs, classes in data_loader:
            inputs, classes = inputs.to(device), classes.to(device)

            out_samples = torch.zeros(model_var_samples, latent_var_samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(model_var_samples, latent_var_samples,
                                    inputs.size(0), model.latent_dim).to(device)

            # sampling the model, then z and classifying
            for k in range(model_var_samples):
                encoded_mu, encoded_std,_ = model.encode(inputs)

                for i in range(latent_var_samples):
                    z = model.reparameterize(encoded_mu, encoded_std)
                    z_samples[k][i] = z

                    cl = model.classifier(z)
                    out = torch.nn.functional.softmax(cl, dim=1)
                    out_samples[k][i] = out

            # calculate the mean and std. Only removes a dummy dimension if number of variational samples is set to one.
            out_mean = torch.mean(torch.mean(out_samples, dim=0), dim=0)
            # preds = out_mean.cpu().data
            all_preds.extend(out_mean.cpu().data)
            #all_indices.extend(indexes)
            if model_var_samples > 1:
                out_std = torch.std(torch.mean(out_samples, dim=0), dim=0)
            else:
                out_std = torch.squeeze(torch.std(out_samples, dim=1))
            zs_mean = torch.mean(torch.mean(z_samples, dim=0), dim=0)
            
            # calculate entropy for the means of samples: - sum pc*log(pc)
            eps = 1e-10
            out_entropy.append(- torch.sum(out_mean*torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

            # In contrast to the eval_dataset function, there is no split into correct or false values as the dataset
            # is unknown.
            for i in range(inputs.size(0)):
                idx = torch.argmax(out_mean[i]).item()
                out_mus[idx].append(out_mean[i][idx].item())
                out_sigmas[idx].append(out_std[i][idx].item())
                encoded_mus[idx].append(encoded_mu[i].data)
                encoded_sigmas[idx].append(encoded_std[i].data)
                zs[idx].append(zs_mean[i].data)
                #collect_indexes_per_class[idx].append(indexes[i].item())

    # stack latent activations into a tensor
    for i in range(len(encoded_mus)):
        if len(encoded_mus[i]) > 0:
            encoded_mus[i] = torch.stack(encoded_mus[i], dim=0)
            encoded_sigmas[i] = torch.stack(encoded_sigmas[i], dim=0)
            zs[i] = torch.stack(zs[i], dim=0)

    out_entropy = np.concatenate(out_entropy).ravel().tolist()

    d={"encoded_mus": encoded_mus, "encoded_sigmas": encoded_sigmas,
            "out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs,
            "out_entropy": out_entropy,'all_preds':all_preds,    "all_indices":all_indices}

    # with open('outfile', 'wb') as fp:pickle.dump(d, fp)

    all_preds=[]
    for i in range(0,len(d['all_preds'])):
        idx=torch.argmax(d['all_preds'][i]).item()
        all_preds.append(d['all_preds'][i][idx])#.data.item())
    all_preds
    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    # need to multiply by -1 to be able to use torch.topk 
    all_preds *= -1

    # select the points which the discriminator things are the most likely to be unlabeled
    _, querry_indices = torch.topk(all_preds, int(args.budget))
    #querry_pool_indices = np.asarray(all_indices)[querry_indices]

    # Return a dictionary of stored values.
    # with open('outfile', 'wb') as fp:pickle.dump(d, fp)
    return {"encoded_mus": encoded_mus, "encoded_sigmas": encoded_sigmas,
            "out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs,
            "out_entropy": out_entropy,'all_preds':all_preds,
            "all_indices":all_indices,
            # 'querry_pool_indices':querry_pool_indices,
            'collect_indexes_per_class':collect_indexes_per_class}