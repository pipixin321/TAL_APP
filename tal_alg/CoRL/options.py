import os
import argparse
import shutil
import yaml
import numpy as np
import utils

#define class dict
_CLASS_NAME = {
    'BEOID':['rinse_cup', 'move_rest', 'take_cup', 'open_door', 'move_seat', 
            'pull_drawer', 'insert_wire', 'place_tape', 'plug_plug', 'pour_spoon', 
            'pull-out_weight-pin', 'switch-on_socket', 'fill_cup', 'push_rowing-machine', 
            'press_button', 'pick-up_cup', 'insert_weight-pin', 'insert_foot', 'scoop_spoon',
            'take_spoon', 'turn_tap', 'pick-up_plug', 'hold-down_button', 'rotate_weight-setting',
            'open_jar', 'let-go_rowing-machine', 'put_jar', 'pull_rowing-machine', 'stir_spoon',
            'put_cup', 'scan_card-reader', 'push_drawer', 'pick-up_jar', 'pick-up_tape'],

    'GTEA':['stir', 'open', 'put', 'close', 'take', 'pour', 'scoop'],

    "THUMOS14":['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
              'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
              'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
              'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
              'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'],

    "ActivityNet13":['Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle',
                    'BMX', 'Baking cookies', 'Ballet', 'Bathing dog', 'Baton twirling',
                    'Beach soccer', 'Beer pong', 'Belly dance', 'Blow-drying hair', 'Blowing leaves',
                    'Braiding hair', 'Breakdancing', 'Brushing hair', 'Brushing teeth', 'Building sandcastles',
                    'Bullfighting', 'Bungee jumping', 'Calf roping', 'Camel ride', 'Canoeing', 'Capoeira',
                    'Carving jack-o-lanterns', 'Changing car wheel', 'Cheerleading', 'Chopping wood',
                    'Clean and jerk', 'Cleaning shoes', 'Cleaning sink', 'Cleaning windows', 'Clipping cat claws',
                    'Cricket', 'Croquet', 'Cumbia', 'Curling', 'Cutting the grass', 'Decorating the Christmas tree',
                    'Disc dog', 'Discus throw', 'Dodgeball', 'Doing a powerbomb', 'Doing crunches', 'Doing fencing',
                    'Doing karate', 'Doing kickboxing', 'Doing motocross', 'Doing nails', 'Doing step aerobics',
                    'Drinking beer', 'Drinking coffee', 'Drum corps', 'Elliptical trainer', 'Fixing bicycle', 'Fixing the roof',
                    'Fun sliding down', 'Futsal', 'Gargling mouthwash', 'Getting a haircut', 'Getting a piercing', 'Getting a tattoo',
                    'Grooming dog', 'Grooming horse', 'Hammer throw', 'Hand car wash', 'Hand washing clothes', 'Hanging wallpaper',
                    'Having an ice cream', 'High jump', 'Hitting a pinata', 'Hopscotch', 'Horseback riding', 'Hula hoop',
                    'Hurling', 'Ice fishing', 'Installing carpet', 'Ironing clothes', 'Javelin throw', 'Kayaking', 'Kite flying',
                    'Kneeling', 'Knitting', 'Laying tile', 'Layup drill in basketball', 'Long jump', 'Longboarding',
                    'Making a cake', 'Making a lemonade', 'Making a sandwich', 'Making an omelette', 'Mixing drinks',
                    'Mooping floor', 'Mowing the lawn', 'Paintball', 'Painting', 'Painting fence', 'Painting furniture',
                    'Peeling potatoes', 'Ping-pong', 'Plastering', 'Plataform diving', 'Playing accordion', 'Playing badminton',
                    'Playing bagpipes', 'Playing beach volleyball', 'Playing blackjack', 'Playing congas', 'Playing drums',
                    'Playing field hockey', 'Playing flauta', 'Playing guitarra', 'Playing harmonica', 'Playing ice hockey',
                    'Playing kickball', 'Playing lacrosse', 'Playing piano', 'Playing polo', 'Playing pool', 'Playing racquetball',
                    'Playing rubik cube', 'Playing saxophone', 'Playing squash', 'Playing ten pins', 'Playing violin',
                    'Playing water polo', 'Pole vault', 'Polishing forniture', 'Polishing shoes', 'Powerbocking', 'Preparing pasta',
                    'Preparing salad', 'Putting in contact lenses', 'Putting on makeup', 'Putting on shoes', 'Rafting',
                    'Raking leaves', 'Removing curlers', 'Removing ice from car', 'Riding bumper cars', 'River tubing',
                    'Rock climbing', 'Rock-paper-scissors', 'Rollerblading', 'Roof shingle removal', 'Rope skipping',
                    'Running a marathon', 'Sailing', 'Scuba diving', 'Sharpening knives', 'Shaving', 'Shaving legs',
                    'Shot put', 'Shoveling snow', 'Shuffleboard', 'Skateboarding', 'Skiing', 'Slacklining',
                    'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Snow tubing', 'Snowboarding', 'Spinning',
                    'Spread mulch','Springboard diving', 'Starting a campfire', 'Sumo', 'Surfing', 'Swimming',
                    'Swinging at the playground', 'Table soccer','Tai chi', 'Tango', 'Tennis serve with ball bouncing',
                    'Throwing darts', 'Trimming branches or hedges', 'Triple jump', 'Tug of war', 'Tumbling', 'Using parallel bars',
                    'Using the balance beam', 'Using the monkey bar', 'Using the pommel horse', 'Using the rowing machine',
                    'Using uneven bars', 'Vacuuming floor', 'Volleyball', 'Wakeboarding', 'Walking the dog', 'Washing dishes',
                    'Washing face', 'Washing hands', 'Waterskiing', 'Waxing skis', 'Welding', 'Windsurfing', 'Wrapping presents',
                    'Zumba']
}


def parse_args():
    parser=argparse.ArgumentParser("This is a baseline of Weakly_supervised Temporal Action Localization")

    # root="./"
    root='/mnt/data1/zhx/TAL_APP/tal_alg/CoRL/'
    parser.add_argument('--seed', type=int, default=0, help='random seed (-1 for no manual seed)')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--ckpt_path', type=str, default=root+"ckpt")
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--hyp', type=str, default='./cfgs/THUMOS14/thumos_swin_tiny.yaml', help='hyperparameters path')
    args=parser.parse_args() 

    #dataset specific hyper-params
    # if args.hyp=="thu":
    #     # args.hyp=root+'cfgs/THUMOS14/thumos_hyp.yaml'
    #     args.hyp=root+'cfgs/THUMOS14/thumos_swin_tiny.yaml'
    # elif args.hyp=="act":
    #     args.hyp=root+'cfgs/ActivityNet13/activitynet_hyp.yaml'
    # elif args.hyp=="beoid":
    #     args.hyp=root+'cfgs/BEOID/beoid_hyp.yaml'
    # elif args.hyp=="gtea":
    #     args.hyp=root+'cfgs/GTEA/gtea_hyp.yaml'
    # else:
    #     AssertionError("Invalid dataset")
    args.hyp=utils.check_file(args.hyp)
    with open(args.hyp) as f:
        args.hyp=yaml.load(f,Loader=yaml.FullLoader)

    #path to save models\outputs\logs
    args.stage=args.hyp['stage']
    args.run_info=args.hyp["run_info"]
    args.model_path=os.path.join(args.ckpt_path,args.hyp["dataset"],"models",args.run_info)
    args.output_path=os.path.join(args.ckpt_path,args.hyp["dataset"],"outputs",args.run_info)
    args.log_path=os.path.join(args.ckpt_path,args.hyp["dataset"],"logs",args.run_info)
    
    #feature info
    args.dataset=args.hyp["dataset"]
    args.data_path=args.hyp["data_path"]
    args.feat_path=args.hyp["feat_path"]
    args.modal=args.hyp["modal"]
    args.supervision=args.hyp["supervision"]
    args.frames_per_sec=args.hyp["frames_per_sec"]
    args.segment_frames_num=args.hyp["segment_frames_num"]
    args.num_segments=args.hyp["num_segments"]
    args.feature_dim=args.hyp["feature_dim"]
    args.scale=args.hyp["scale"]
    
    #training params
    args.gpu=args.hyp["gpu"]
    args.num_workers=args.hyp["num_workers"]
    args.num_iters=args.hyp["num_iters"]
    args.test_iter=args.hyp["test_iter"]
    args.pseudo_freq=args.hyp["pseudo_freq"]
    args.batch_size=args.hyp["batch_size"]
    args.lr=args.hyp["lr"]
    args.weight_decay=args.hyp["weight_decay"]
    args.dropout=args.hyp["dropout"]
    args.transformer_args=args.hyp["transformer_args"]

    #hyparam
    args.r_act=args.hyp["r_act"]
    args.class_thresh=args.hyp["class_thresh"]
    args.act_thresh_cas=eval(args.hyp["act_thresh_cas"])
    args.act_thresh_agnostic=eval(args.hyp["act_thresh_agnostic"])
    args.lambdas=eval(args.hyp["lambdas"])
    args.nms_thresh=args.hyp["nms_thresh"]
    args.nms_alpha=args.hyp["nms_alpha"]
    args.tIoU_thresh=eval(args.hyp["tIoU_thresh"])
    args._lambda=args.hyp["_lambda"]
    args.gamma=args.hyp["gamma"]
    args.blocked_videos=args.hyp["blocked_videos"]

    #component control
    args.manual_id=args.hyp["manual_id"]
    args.BaS=eval(args.hyp["BaS"])
    args.part_topk=eval(args.hyp["part_topk"])
    args.agnostic_inf=eval(args.hyp["agnostic_inf"])
    args.count_loss=eval(args.hyp["count_loss"])

    args.class_name_lst=_CLASS_NAME[args.dataset]
    args.action_cls_num=len(args.class_name_lst)
    
    

    if args.dataset=="ActivityNet13":
        args.test_info={"step": [], "test_acc": [],
                    "average_mAP[0.5:0.95]": [],
                    "mAP@0.50": [], "mAP@0.55": [], "mAP@0.60": [], "mAP@0.65": [], "mAP@0.70": [],
                    "mAP@0.75": [], "mAP@0.80": [], "mAP@0.85": [], "mAP@0.90": [],"mAP@0.95": []}
        
    else:
        args.test_info={"step": [], "test_acc": [],
                    "average_mAP[0.1:0.7]": [], "average_mAP[0.1:0.5]": [], "average_mAP[0.3:0.7]": [],
                    "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], "mAP@0.4": [],
                    "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": []}

    print(args.hyp)
    return init_args(args)

def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args


if __name__ == "__main__":
    args=parse_args()
    print(args)

