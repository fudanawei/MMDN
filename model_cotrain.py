import numpy as np
from sklearn.metrics import accuracy_score, compare_mse

# accuracy evaluation
def c_evaluate(y_actual, y_predicted, graylevel=2):
    accuracy = []
    for i in range(0, y_actual.shape[0]):
        if graylevel > 2:
            a = compare_mse(y_actual[i], y_predicted[i])
        else:
            a = accuracy_score(y_actual[i], y_predicted[i]) * 100
        accuracy.append(a)
    accuracy_diffimage = abs(y_actual-y_predicted).mean(axis=0)

    return accuracy, np.mean(accuracy), accuracy_diffimage

# TODO: Confidence level --> regularization: TV minimization; discriminator; 
def confidenceFactor_V2(arr, graylevel=2): 
    W = []
    for i in range(0, arr.shape[0]):
        img = np.clip(arr[i], 0, 1) * (graylevel-1)
        img_t = (0.5 + img).astype(np.uint8)
        w = np.sum(1 - np.abs(img - img_t) * 2) / arr.shape[-1]  # range: 0-1        
        W.append(w)
    meanW = np.mean(W)
    return W, meanW

# return confidence level, 0.1~1.0
def confidenceFactor(arr):
    median = 0.5
    confList = []
    for i in range(0, arr.shape[0]):
        conf = np.sum(np.abs(arr[i] - median)*1.8 + 0.1) / arr.shape[-1]
        confList.append(conf)
    meanConf = np.mean(confList)
    return confList, meanConf

# confidence based weight, TODO: check!!
def weightFactor(conf1, conf2, conf3):
    # w1 = np.exp(10 * (2 - conf2 - conf3) / 3)
    # w2 = np.exp(10 * (2 - conf1 - conf3) / 3)
    # w3 = np.exp(10 * (2 - conf1 - conf2) / 3)
    w1 = np.exp(10 * (2 - conf2 - conf3) / (3 - conf1 - conf2 - conf3))
    w2 = np.exp(10 * (2 - conf1 - conf3) / (3 - conf1 - conf2 - conf3))
    w3 = np.exp(10 * (2 - conf1 - conf2) / (3 - conf1 - conf2 - conf3))
    return w1, w2, w3

# squared hellinger distance
def hellinger(p, q):
    dist = np.sum(np.square(np.sqrt(p) - np.sqrt(q))) / 2
    return dist

# uniform division
def binarize(inarr, graylevel=2):
    outarr = np.clip(inarr,0,1)
    outarr = (0.5+outarr*(graylevel-1)).astype(np.uint8) / (graylevel-1) 
    if graylevel > 2:
        return outarr.astype(np.float16)
    else:
        return outarr.astype(np.uint8)

def binarize_byThreshold(inarr, graylevel=2, threshold=[0.5]):
    assert len(threshold) == graylevel-1
    outarr = np.clip(inarr,0,1)
    for i in range(graylevel-1):
        th = threshold[-i-1]
        outarr[outarr>th] = graylevel-i-1
    outarr = (outarr).astype(np.uint8) # digitize: 0~graylevel-1 
    return outarr

def co_train_evluation(model1,model2,model3,x_fortest,y_fortest,pre_M1,pre_M2,pre_M3,graylevel=2):
    # model prediction
    predict1 = model1.predict(x_fortest)
    predict2 = model2.predict(x_fortest)
    predict3 = model3.predict(x_fortest)

    # cache grayscale predictions for seperate models 
    pre_M1.append(predict1)
    pre_M2.append(predict2)
    pre_M3.append(predict3)

    # confidence-based weighted ensemble of predicted images
    conf1 = confidenceFactor(predict1)[1]
    conf2 = confidenceFactor(predict2)[1]
    conf3 = confidenceFactor(predict3)[1]
    np.set_printoptions(precision=4)
    print('confidence:', np.array([conf1, conf2, conf3]))
    
    w1, w2, w3 = weightFactor(conf1, conf2, conf3)
    np.set_printoptions(precision=4)
    print('weight: ', np.array([w1, w2, w3]))

    predsb = (w1 * predict1 + w2 * predict2 + w3 * predict3) / (w1 + w2 + w3)
    predsb = binarize(predsb, graylevel)
    evaluationC4 = c_evaluate(y_fortest, predsb)
    
    # evaluation of binarized single-model predictions
    predsb1 = binarize(predict1, graylevel)
    predsb2 = binarize(predict2, graylevel)
    predsb3 = binarize(predict3, graylevel)
    evaluationC1 = c_evaluate(y_fortest, predsb1)
    evaluationC2 = c_evaluate(y_fortest, predsb2)
    evaluationC3 = c_evaluate(y_fortest, predsb3)
    
    return predsb,evaluationC1,evaluationC2,evaluationC3,evaluationC4,conf1,conf2
   

def single_train_evluation(model,x_fortest,y_fortest,pre_M):
    # model prediction
    predict = model.predict(x_fortest)

    # cache grayscale predictions for seperate models 
    pre_M.append(predict)
    
    predsb = binarize(predict, graylevel=2)
    evaluationC = c_evaluate(y_fortest, predsb)
    
    return predsb,evaluationC