import numpy as np
import soundfile as sf
import glob, random, os, torch
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderASR
from pystoi import stoi
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem


def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy

    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def moo1(label='',  ## task1, use STOI
    random_num = 10,
    algorithm = NSGA2(pop_size=100),
    n_gen = 20):

    # define verification
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

    wave_root = '../dataset/voxceleb1/wav-8k/'  # root path of your dataset
    save_root = '../data/' + label + '/'  # root path of anonymized audio
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    ## randomly load train-set
    wave_paths = glob.glob(os.path.join(wave_root, '*/*/*.wav'))


    # save result to file 
    save_file = '' + label + '.txt'  # add your save path to save each iteration result

    class MicPro(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=3, n_obj=2, xl=0, xu=2)
        
        def _evaluate(self, x, out, *args, **kwargs):
            ## write params to txt file
            f = open('param_moo.txt','w') 
            for i in range(3):
                print(x[i], file=f)
            f.close()
        
            ## compile G.729 codec
            os.chdir('g729/')
            os.system('rm *.o')
            os.system('make -f coder.mak')
            os.system('make -f decoder.mak')
            os.system('rm *.o')

            ## calculate obj value
            index = 0
            value1 = 0
            value2 = 0
            train_set = random.sample(wave_paths, random_num) 
            for i in range(random_num):
                orig_path = train_set[index]
                save_path = save_root + os.path.split(orig_path)[1].split('.')[0]
                bin_path = save_path + 'g729.bin'
                pcm_path = save_path + 'g729.pcm'
                wav_path = save_path + 'g729.wav'
                pcm_path_16k = save_path + 'g729_16k.pcm'
                index += 1

                os.system('g729/coder %s %s' %(orig_path, bin_path))
                os.system('g729/decoder %s %s' %(bin_path, pcm_path))
                os.system('ffmpeg -y -ar 8000 -ac 1 -f s16le -i %s -ar 16000 -ac 1 -f s16le %s' %(pcm_path, pcm_path_16k))
                os.system('ffmpeg -f s16le -v 8 -y -ar 16000 -ac 1 -i %s %s' %(pcm_path_16k, wav_path))

                orig_data, sr = sf.read(orig_path.replace('wav-8k','wav'))
                anon_data, sr = sf.read(wav_path)

                # calculate ASV cosine distance
                score_asv, _ = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))  # 0~1
                score_asv = float(score_asv[0])
                value1 += score_asv

                min_len = np.min([len(orig_data), len(anon_data)])
                # get stoi
                score_stoi = stoi(orig_data[0:min_len], anon_data[0:min_len], sr)  # 0-1
                value2 += 1-score_stoi

                # calculate loss
                
                f = open(save_file,'a')
                print("%d wave_path:%s ; asv_score:%.6f ; stoi:%.6f; value1:%.6f; value2:%.6f" %(index, orig_path, score_asv, score_stoi, value1, value2), file=f)
                print(x , file=f)
                # print('-'*20, file=f)
                f.close()
            out['F'] = [value1, value2]

    problem = MicPro()

    res = minimize(problem=problem,
                    algorithm=algorithm,
                    termination=('n_gen', n_gen),
                    seed=1,
                    save_history=True,
                    verbose=True
                    )
    X=res.X
    F=res.F
    f = open(save_file,'a')
    print(X, file=f)
    print(F, file=f)
    f.close()




def moo2(label='',  ## task2, use WER
    random_num = 10,
    algorithm = NSGA2(pop_size=100),
    n_gen = 20):

    # define verification
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    asr_model = EncoderASR.from_hparams(source='speechbrain/asr-wav2vec2-librispeech', savedir='pretrained_models/asr-wav2vec2-librispeech')

    wave_root = '../dataset/voxceleb1/wav-8k/'  # root path of your dataset
    save_root = '../data/' + label + '/'  # root path of anonymized audio
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    ## randomly load train-set
    wave_paths = glob.glob(os.path.join(wave_root, '*/*/*.wav'))
    

    # save result to file 
    save_file = '' + label + '.txt'  # add your save path to save each iteration result

    class MicPro(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=3, n_obj=2, xl=0, xu=2)
        
        def _evaluate(self, x, out, *args, **kwargs):
            ## write params to txt file
            f = open('param_moo.txt','w')
            for i in range(3):
                print(x[i], file=f)
            f.close()
        
            ## compile G.729 codec
            os.chdir('g729/')
            os.system('rm *.o')
            os.system('make -f coder.mak')
            os.system('make -f decoder.mak')
            os.system('rm *.o')

            ## calculate obj value
            index = 0
            value1 = 0
            value2 = 0
            wrongnum = 0
            wordnum = 0
            train_set = random.sample(wave_paths, random_num) 
            for i in range(random_num):
                orig_path = train_set[index]
                save_path = save_root + os.path.split(orig_path)[1].split('.')[0]
                bin_path = save_path + 'g729.bin'
                pcm_path = save_path + 'g729.pcm'
                wav_path = save_path + 'g729.wav'
                pcm_path_16k = save_path + 'g729_16k.pcm'
                index += 1

                os.system('g729/coder %s %s' %(orig_path, bin_path))
                os.system('g729/decoder %s %s' %(bin_path, pcm_path))
                os.system('ffmpeg -y -ar 8000 -ac 1 -f s16le -i %s -ar 16000 -ac 1 -f s16le %s' %(pcm_path, pcm_path_16k))
                os.system('ffmpeg -f s16le -v 8 -y -ar 16000 -ac 1 -i %s %s' %(pcm_path_16k, wav_path))

                orig_data, sr = sf.read(orig_path.replace('wav-8k','wav'))
                anon_data, sr = sf.read(wav_path)

                # calculate ASV cosine distance
                score_asv, _ = verification.verify_batch(torch.tensor(orig_data),torch.tensor(anon_data))  # 0~1
                score_asv = float(score_asv[0])
                value1 += score_asv

                rel_length = torch.tensor([1.0])
                result_real, _ = asr_model.transcribe_batch(torch.tensor(orig_data).unsqueeze(0), rel_length)
                result_fake, _ = asr_model.transcribe_batch(torch.tensor(anon_data).unsqueeze(0), rel_length)
                result_real = result_real[0]
                result_fake = result_fake[0]
                number = wer(result_real.split(), result_fake.split())
                wrongnum += number
                wordnum += len(result_real.split())
                score_asr = (float)(wrongnum/wordnum)
                value2 = score_asr

                # calculate loss
                f = open(save_file,'a')
                print("%d wave_path:%s ; asv_score:%.6f ; asr_score:%.6f; value1:%.6f; value2:%.6f" %(index, orig_path, score_asv, score_asr, value1, value2), file=f)
                print(x , file=f)
                # print('-'*20, file=f)
                f.close()
            out['F'] = [value1, value2]

    problem = MicPro()

    res = minimize(problem=problem,
                    algorithm=algorithm,
                    termination=('n_gen', n_gen),
                    seed=1,
                    verbose=True,
                    save_history=True
                    )
    X=res.X
    F=res.F
    f = open(save_file,'a')
    print(X, file=f)
    print(F, file=f)
    f.close()


if __name__ == '__main__':
    moo1(label='')  # task1
    # moo2(label='')  # task2
    pass

