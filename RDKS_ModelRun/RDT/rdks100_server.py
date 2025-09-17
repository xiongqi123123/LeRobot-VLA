# Copyright (c) 2025, Cauchy WuChao, D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from server_client import Client
from BPU_RDT_Policy import *
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpu_rdt_path', type=str, default='./BPU_RDT_Policy/', help='') 
    # example: $ tree BPU_RDT_Policy
    # .
    # |-- base.yaml
    # |-- bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm
    # |-- rdt_dit.hbm
    # |-- rdt_img_adaptor.hbm
    # |-- rdt_lang_adaptor.onnx
    # |-- rdt_state_adaptor_1x1x256.onnx
    # `-- rdt_state_adaptor_1x64x256.onnx
    parser.add_argument('--host', type=str, default='10.112.20.37', help='')
    parser.add_argument('--port', type=int, default=50023, help='')
    parser.add_argument('--ctrl_freq', type=int, default=25, help="")
    parser.add_argument('--left_arm_dim', type=int, default=6, help="")
    parser.add_argument('--right_arm_dim', type=int, default=6, help="")
    opt = parser.parse_args()


    with open(os.path.join(opt.bpu_rdt_path, "base.yaml"), "r") as fp:
        config_base_yaml = yaml.safe_load(fp)
    config_base_yaml["arm_dim"] = {"left_arm_dim": opt.left_arm_dim, "right_arm_dim": opt.right_arm_dim}
    config_base_yaml['ctrl_freq'] = opt.ctrl_freq

    # bpu_model = BPU_RDT_Policy(opt.bpu_rdt_path, config_base_yaml)
    bpu_model = BPU_RDT_Policy(opt.bpu_rdt_path, config_base_yaml, SERVER_URL = 'http://10.64.60.208:5000/process')

    logger.info("BPU RDT model initialized")

    client = Client(host=opt.host, port=opt.port)  


    while True:
        logger.info("Wait for Received. ")
        none_cnt = 0
        while True: 
            data = client.receive()
            if data is None:
                print(".", end=" ", flush=True)
                sleep(0.1)
                none_cnt += 1
                if none_cnt > 50:
                    client.close()
                    del client
                    client = Client(host=opt.host, port=opt.port)  
                    break
            else:
                break
        try:
            if 'flag' not in data.keys():
                logger.info("\'flag\' not in data.keys()")
                continue
            if data['flag'] == 'step':
                actions = bpu_model.step([data['imgs_0'], 
                                        data['imgs_1'], 
                                        data['imgs_2'], 
                                        data['imgs_3'], 
                                        data['imgs_4'], 
                                        data['imgs_5']], 
                                        data['joints'])
                client.send({
                    'actions': actions
                })
            elif data['flag'] == 'set_lang_condition':
                bpu_model.set_lang_condition(data['instruction'])
                client.send({
                    'msg': "OK"
                })
            else:
                logger.info(f"Unknow flag {data['flag']}")
            continue

        except Exception as e:
            logger.info(f"ERROR: {str(e)}")
            import traceback
            logger.info(f"traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()