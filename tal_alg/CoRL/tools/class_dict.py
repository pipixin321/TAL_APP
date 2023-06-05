
class class_dict:
    def __init__(self,dataset='thu'):
        self.class_name = {
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
        if dataset=='thu':
            self.class_name_lst=self.class_name['THUMOS14']
        elif dataset=='act':
            self.class_name_lst=self.class_name['ActivityNet13']
        elif dataset == 'beoid':
            self.class_name_lst=self.class_name['BEOID']
        elif dataset == 'gtea':
            self.class_name_lst=self.class_name['GTEA']
        self.class_dict={i:v for i,v in enumerate(self.class_name_lst)}