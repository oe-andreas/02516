## ToDo Object Detection
- [] Visualize some images with GT boxes, SS boxes and EB boxes, to understand the poor recall/MABO.
- [] Maybe we should lower the threshold for when a proposal box is considered to match a GT box in `bounding_box_recall_and_mabo`
- [] Play around with input parameters for edge boxes and SS, since our recall/MABO is not very impressive right now. Most easily done like in the section 'edge boxes' in test_oe.ipynb (should probably be moved to HPC for speed though). Should probably reduce to a subset of images so that different parameters can easily be tested
- [] Create a training loop. Note that loss should combine regression and classification loss.
- [] Implement NMS for part 3
- [] Consider making a flow chart of everything that goes on at test time (images goes into SS -> bboxes... Each bbox goes through Network... boxes are adjusted according to bbox regression... NMS... etc)


## Questions
- What recall/MABO is acceptable?
- Is overall stratified sampling ok, or should it be per-batch?