from module.dataloaders.loader_test_time import Dataloader_test_time
import torch
from module.models.efficientnet import EfficientNetWithBBox
from module.processing.non_maximum_suppression import non_maximum_suppression
from module.metrics.AP import AP
from tqdm import tqdm  # For progress bar
import pickle
from datetime import datetime



#test pickle to avoid running the whole thing
pickle.dump(['test'], open("dumps/test.pkl", "wb"))
max_proposals = 1500
data_loader = Dataloader_test_time(train = 'test', shuffle_proposals_in_im=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'Trained_models/frederik_good_model.pth'
model_name = 'efficientnet_b0'



model = EfficientNetWithBBox(model_name, pretrained=False, num_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)


all_gt_bboxs = []
all_positive_proposal_bboxs = []
all_positive_proposals_probs = []

for Xs, bboxs, gt_bboxs in tqdm(data_loader, total=len(data_loader)):
    
    Xs = Xs.to(device)
    
    proposals = Xs[:max_proposals]
    
    bboxs = bboxs[:max_proposals]
    
    class_score_logits, t_vals = model(proposals)
    
    probs = torch.sigmoid(class_score_logits).squeeze().detach()

    
    #extract only positive examples
    proposals = proposals[probs > 0.5]
    probs = probs[probs > 0.5]
    
    boxes_w_probs = list(zip(bboxs, probs))

    #NMS
    _, selected_indeces = non_maximum_suppression(boxes_w_probs, 0.5, return_indeces=True)

    
    #get the proposals that were selected
    proposals = proposals[selected_indeces]
    probs = probs[selected_indeces]
    bboxs = bboxs[selected_indeces]
    
    all_gt_bboxs.extend(gt_bboxs)
    all_positive_proposal_bboxs.extend(bboxs)
    all_positive_proposals_probs.extend(probs)
    


current_time = datetime.now().strftime("%Y%m%d_%H%M")
pickle.dump((all_gt_bboxs, all_positive_proposal_bboxs, all_positive_proposals_probs), open(f"dumps/ap_input_{current_time}.pkl", "wb"))


ap = AP(all_gt_bboxs, all_positive_proposal_bboxs, all_positive_proposals_probs)
print(ap)