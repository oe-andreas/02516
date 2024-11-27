from module.dataloaders.loader_test_time import Dataloader_test_time
import torch
from module.models.efficientnet import EfficientNetWithBBox
from module.processing.non_maximum_suppression import non_maximum_suppression
from module.metrics.AP import AP
from tqdm import tqdm  # For progress bar
import pickle
from datetime import datetime

#UPDATE THESE
model_path = 'Trained_models/b0_model_20241117_0044.pth'
model_name = 'efficientnet_b0'
max_proposals = 1500
data_loader = Dataloader_test_time(train = 'train', shuffle_proposals_in_im=True)


#test pickle to avoid running the whole thing
pickle.dump(['test'], open("dumps/test.pkl", "wb"))

#start script stuff
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')
model = EfficientNetWithBBox(model_name, pretrained=False, num_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print(f"Total memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3}")
print(f"Total memory allocated after loading model: {torch.cuda.memory_allocated(device) / 1024**3}")


all_gt_bboxs_w_ids = []
#all_positive_proposal_bboxs = []
all_prop_boxes_w_tvals_probs_and_ids = []

#all_positive_adjusted_bboxs = []
#all_positive_proposals_probs = []



for Xs, bboxs, gt_bboxs, id in tqdm(data_loader, total=len(data_loader)):
    
    print(f'Id = {id}')

    Xs = Xs.to(device)
    
    #check memory size
    #print(f"Memory allocated for Xs: {Xs.element_size() * Xs.nelement() / 1024**3}")
    #print(f"Total memory allocated: {torch.cuda.memory_allocated(device) / 1024**3}")
    #print(f"Total memory reserved: {torch.cuda.memory_reserved(device) / 1024**3}")
    
    proposals = Xs[:max_proposals]
    #print("Managed to define proposals")
    
    bboxs = bboxs[:max_proposals]
    #print("Managed to define bboxs")
    
    print(f"Considering {len(proposals)} proposals")
    
    with torch.no_grad(): #save memory usage by not saving gradients
        class_score_logits, t_vals = model(proposals)
    #print("Managed to run model")
    
    probs = torch.sigmoid(class_score_logits).squeeze().detach().cpu()
    #print("Managed to calc probs")

    
    #extract only positive examples
    bboxs_pos = bboxs[probs > 0.5]
    #tvals = t_vals[probs > 0.5]
    #probs = probs[probs > 0.5]
    
    
    
    
    
    print(f"Found {len(bboxs_pos)} positive proposals")
    
    #boxes_w_probs = list(zip(bboxs, probs))

    #NMS
    #_, selected_indeces = non_maximum_suppression(boxes_w_probs, 0.5, return_indeces=True)

    #print("Managed to NMS")
    
    #get the proposals that were selected
    #probs = probs[selected_indeces]
    #bboxs = bboxs[selected_indeces]
    
    #print("Managed to do second masking")
    
    #all_gt_bboxs.append((gt_bboxs, id))
    all_gt_bboxs_w_ids.append((gt_bboxs, id))
    all_prop_boxes_w_tvals_probs_and_ids.append((bboxs.cpu(), t_vals.cpu(), probs.cpu(), id))
    #all_positive_proposal_bboxs.append((zip(bboxs.cpu(), probs.cpu()), id))
    #all_positive_proposals_probs.extend((probs.cpu(), i))
    
    #print("Managed to extend lists")
    
    #print(f"Memory allocated for Xs end: {Xs.element_size() * Xs.nelement() / 1024**3}")
    #print(f"Total memory allocated end: {torch.cuda.memory_allocated(device) / 1024**3}")
    #print(f"Total memory reserved end: {torch.cuda.memory_reserved(device) / 1024**3}")
    
    
    


current_time = datetime.now().strftime("%Y%m%d_%H%M")
pickle.dump((all_gt_bboxs_w_ids, all_prop_boxes_w_tvals_probs_and_ids), open(f"dumps/ap_input_{current_time}.pkl", "wb"))


#ap = AP(all_gt_bboxs, all_positive_proposal_bboxs, all_positive_proposals_probs)
#print(ap)