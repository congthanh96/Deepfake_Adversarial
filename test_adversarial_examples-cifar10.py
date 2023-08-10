import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import CIFAR_target_net



if __name__ == "__main__":
    
    use_cuda=True
    image_nc=3
    batch_size = 1

    gen_input_nc = image_nc

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    device = 'cpu'

    # load the pretrained model
    #pretrained_model = "./CIFAR_target_model.pth"

    #target_model = CIFAR_target_net().to(device)

    #target_model.load_state_dict(torch.load(pretrained_model))
    #target_model.eval()

    MODEL_NAME = "cifar10_repvgg_a2"
    target_model = torch.hub.load("chenyaofo/pytorch-cifar-models", MODEL_NAME, pretrained=True)
    target_model = target_model.to(device)
    target_model.eval()

    # load the generator of adversarial examples
    #pretrained_generator_path = './models/netG.pth.tar.38'
    #pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    #pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    #pretrained_G.eval()
    pretrained_generator_path = './models/netG.pth.tar.600'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    # test adversarial examples in CIFAR10 training dataset
    # cifar10_dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=transforms.ToTensor(), download=True)

    # train_dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    # num_correct = 0
    # num_all = 0
    # for i, data in enumerate(train_dataloader, 0):
    #     test_img, test_label = data
    #     test_img, test_label = test_img.to(device), test_label.to(device)
    #     perturbation = pretrained_G(test_img)
    #     perturbation = torch.clamp(perturbation, -0.3, 0.3)
    #     adv_img = perturbation + test_img
    #     adv_img = torch.clamp(adv_img, 0, 1)
    #     pred_lab = torch.argmax(target_model(adv_img),1)
    #     num_all += len(test_label)
    #     num_correct += torch.sum(pred_lab==test_label,0)

    # print('cifar10 training dataset:')
    # print('num_examples: ', num_all)
    # print('num_correct: ', num_correct.item())
    # print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/num_all))

    # test adversarial examples in cifar10 testing dataset
    cifar10_dataset_test = torchvision.datasets.CIFAR10('./dataset', train=False, transform=transforms.ToTensor(), download=True)

    test_dataloader = DataLoader(cifar10_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    print("len(test_dataloader): ", len(test_dataloader))
    num_correct = 0
    num_all = 0

    import cv2
    count = 0
    CLASS_NAMES = [ "airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]
    CLASS_NAMES_DICT = {"airplanes": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "cars": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "birds": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "cats": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "deer": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "dogs": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "frogs": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "horses": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "ships": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0} ,
                        "trucks": {"gt": 0, "adv_succeed": 0, "adv_succeed_%": 0}
                    }

    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)

        targeted_rslt = target_model(test_img)
        targeted_lb = torch.argmax(targeted_rslt, 1)


        if (torch.sum(test_label==targeted_lb, 0) == 0):
            continue


        targeted_score = torch.nn.Softmax()(targeted_rslt[0])[targeted_lb[0]].detach().numpy()

        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)

        perturbation_mean = perturbation.mean(dim=[1,2,3])
        perturbation[0] = perturbation[0] - perturbation_mean[0]

        # adv_img = test_img + 0.05*torch.randn(perturbation.shape)
        adv_img = test_img + 0.1*perturbation
        adv_img = torch.clamp(adv_img, 0, 1)

        pred_rslt = target_model(adv_img)
        pred_lab = torch.argmax(pred_rslt, 1)
        pred_scores = torch.nn.Softmax()(pred_rslt[0])[pred_lab[0]].detach().numpy()

        num_all += len(test_label)
        current_correct = torch.sum(pred_lab==test_label,0)
        num_correct += current_correct

        # quick test for batch size = 1
        CLASS_NAMES_DICT[CLASS_NAMES[test_label[0].numpy()]]["gt"] += 1
        if (current_correct.item() != 0):
            continue
        CLASS_NAMES_DICT[CLASS_NAMES[test_label[0].numpy()]]["adv_succeed"] += 1


        cv2.imwrite(f"./preview/id_{count}-0origin -label_{test_label[0].numpy()}-classname_{CLASS_NAMES[test_label[0].numpy()]}.jpg",
                            test_img[0].permute((1,2,0)).numpy()*255)

        cv2.imwrite(f"./preview/id_{count}-1targerted-label_{targeted_lb[0].numpy()}_{targeted_score}-classname_{CLASS_NAMES[targeted_lb[0].numpy()]}.jpg",
                            test_img[0].permute((1,2,0)).numpy()*255)

        cv2.imwrite(f"./preview/id_{count}-2pert.jpg",
                            perturbation[0].permute((1,2,0)).detach().numpy()*255)

        cv2.imwrite(f"./preview/id_{count}-3adv-label_{pred_lab[0].numpy()}_{pred_scores}-classname_{CLASS_NAMES[pred_lab[0].numpy()]}.jpg",
                            adv_img[0].permute((1,2,0)).detach().numpy()*255)

    print('cifar10 test dataset:')
    print('num_examples: ', num_all)
    print('num_correct: ', num_correct.item())
    print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/num_all))
    print('adv succeeded %f\n'%(1-num_correct.item()/num_all))

    for cls_name in CLASS_NAMES:
        CLASS_NAMES_DICT[cls_name]["adv_succeed_%"] = CLASS_NAMES_DICT[cls_name]["adv_succeed"] / CLASS_NAMES_DICT[cls_name]["gt"]

    import json
    CLASS_NAMES_DICT = json.dumps(CLASS_NAMES_DICT, indent=4)
    print(CLASS_NAMES_DICT)
