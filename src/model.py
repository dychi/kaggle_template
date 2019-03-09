from models import ED3D_model
from models.PoseModel import PoseNet
from models.crf import TCN_CRF

def generate_model(args):
    if args.layers == 3:
         model = ED3D_model.ThreeLayer(num_classes=13, model=args.model, train_layer=args.train_layer)
    elif args.layers == 2:
        model = ED3D_model.TwoLayer(num_classes=13, model=args.model, train_layer=args.train_layer)
    else:
        if args.model == 'P3D_flatten':
            model = ED3D_model.EDP3D(num_classes=13)
        elif args.model == 'PoseNet':
            model = PoseNet()
        # Use CRF
        if args.use_crf:
            model = TCN_CRF(n_class=13, model=args.model, train_layer=args.train_layer)
    
    # Setting Finetuning params
    params_to_update = []
    print("Param to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
    
    return model, params_to_update

