from __future__ import print_function

import argparse
import glob
import json
import random
import string
from collections import Counter
from typing import Union, List

import MeCab
import jsonlines
from nltk.translate import bleu

wakati = MeCab.Tagger("-Owakati")

lang_dic = {'telugu': 'te', 'swahili': 'sw', 'thai': 'th', 'finnish': 'fi', 'indonesian': 'id',
            'japanese': 'ja', 'russian': 'ru', 'arabic': 'ar', 'english': 'en', 'bengali': 'bn',
            "korean": "ko", "spanish": "es", "hebrew": "he", "swedish": "sv", "danish": "da", "german": "de",
            "hungarian": "hu", "italian": "it", "khmer": "km", "malay": "ms", "dutch": "nl",
            "norwegian": "no", "portuguese": "pt", "turkish": "tr", "vietnamese": "vi", "french": "fr", "polish": "pl",
            "chinese (simplified)": "zh_cn",  "chinese (hong kong)": 'zh_hk', "chinese (traditional)": "zh_tw"}


def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def load_tydi_answer(tydi_eval_open_domain_data):
    answer_dict = {}
    eval_data = read_jsonlines(tydi_eval_open_domain_data)
    for item in eval_data:
        answer_dict[item["id"]] = item["answers"]
    return answer_dict


def normalize_answer(s):
    # TODO: should we keep those counter removal?
    def remove_counter(text):
        return text.replace("年", "").replace("歳", "").replace("人", "").replace("년", "")

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_counter(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    # print("ground_truth: {}\t prediction: {}\t{}".format(ground_truth, prediction, normalize_answer(prediction) == normalize_answer(ground_truth)))
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# 3. XOR-Full Evaluation
def calculate_f1_em_bleu(dataset, predictions):
    lang_dict = {lang: {"count": 0, "f1": 0, "bleu": 0, "em": 0}
                 for lang in lang_dic.values()}

    for qa in dataset:
        lang = qa["lang"]
        gts = qa["answers"]
        if gts[0] == "No Answer":
            continue
        q_id = qa["id"]
        lang_dict[lang]["count"] += 1
        if q_id not in predictions:
            print("no answers")
            continue
        pred = predictions[q_id]
        if isinstance(gts, str):
            gts = [gts]

        final_gts = []
        # for japanese, we need to tokenize the input as there are no white spaces.
        if lang == "ja":
            for gt in gts:
                gt = wakati.parse(gt)
                final_gts.append(gt)
            final_pred = wakati.parse(pred.replace("・", " ").replace("、", ","))
        else:
            final_gts = gts
            final_pred = pred

        # new_gts = []
        # for gt in final_gts:
        #     if 10 <= len(gt.split(" ")):
        #         new_gts.append(gt)
        # final_gts = new_gts
        # if len(final_gts) == 0:
        #     lang_dict[lang]["count"] -= 1
        #     continue

        lang_dict[lang]["f1"] += metric_max_over_ground_truths(
            f1_score, final_pred, final_gts)
        lang_dict[lang]["bleu"] += bleu(final_gts, pred)
        lang_dict[lang]["em"] += metric_max_over_ground_truths(
            exact_match_score, final_pred, final_gts)
    # finalize scores
    for lang, scores in lang_dict.items():
        if scores["count"] == 0:
            continue
        for score_key in scores:
            if "count" != score_key:
                lang_dict[lang][score_key] = scores[score_key]/scores["count"]
    return lang_dict


def calculate_f1_em_bleu_significant(dataset, predictions):
    outputs = []
    for qa in dataset:
        lang = qa["lang"]
        gts = qa["answers"]
        if gts[0] == "No Answer":
            continue
        q_id = qa["id"]
        if q_id not in predictions:
            print("no answers")
            continue
        pred = predictions[q_id]
        if isinstance(gts, str):
            gts = [gts]

        final_gts = []
        # for japanese, we need to tokenize the input as there are no white spaces.
        if lang == "ja":
            for gt in gts:
                gt = wakati.parse(gt)
                final_gts.append(gt)
            final_pred = wakati.parse(pred.replace("・", " ").replace("、", ","))
        else:
            final_gts = gts
            final_pred = pred

        outputs.append((q_id, metric_max_over_ground_truths(
            f1_score, final_pred, final_gts)))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",
                        default=None, type=str)
    parser.add_argument("--pred_file",
                        default=None, type=str)
    parser.add_argument("--txt_file", action="store_true")
    parser.add_argument("--significant", action="store_true")

    args = parser.parse_args()

    dataset = read_jsonlines(args.data_file)

    # qids = ['-4194574527800227004', '-8534837945449659988', '-49544909738947984', '-1413725843955627404', '-2299841510974346421', '-6642144296187590298', '-3173797870513599511', '-707552382835249129', '-777680077576475779', '-4166868313835054376', '3656450990094647942', '5009708573885932569', '-7777113122145346577', '7135956910125091353', '-7438449460180724010', '-9099010440740722699', '-2607105789597326909', '-1457018415932687185', '8852078010282713990', '4198161683562908654', '7154443653304042982', '-7182236685420999405', '5841137545646136347', '-8676396461317698234', '-2736749878746564523', '-907477793232164936', '4416701920956377562', '7912016739218368308', '-3686633148544147788', '-1792142003460532685', '-601343259055032720', '1480875767130429785', '-5524370408722306136', '-7794185464777822065', '795937211551918144', '3937963433826730062', '8652540446567071794', '2279719630731573582', '-5468287088726908682', '-8862780185516668757', '7601066006322125432', '-8395449471354467275', '-4915659449158888905', '-5795454333476869645', '-4565911940746751956', '-6916806716396464221', '-4356711200244286691', '-1354891130349942872', '4383366077260435539', '6428033608111316307', '-9050504224929755440', '5638058614971690578', '7378506612974018392', '-484153380110401622', '7116667444156391428', '-9005114087681438911', '7075819166056744737', '-2310612763144978658', '-3771301080053535443', '2432685861927278888', '7071808599726455761', '-1091779853143657943', '4610728675765610071', '-4586255834885692095', '1018344720410562580', '-578247977613020425', '8434879232689324272', '1970210989866796851', '5692039763279115737', '-2399396364335826088', '5273406018593685256', '7887725680289525142', '-749738218164430458', '-8627597343898407564', '933694769513133899', '8827559639258790371', '6551710252690197340', '298200800048230018', '7594480950741625774', '3953518350412364336', '-6228868976593966787', '1781071549104033230', '-3077053776453753352', '-1712973995367791240', '6012438969925208015', '5112432004742593961', '4080170451722267299', '6562304557552812836', '-9074153636811957024', '5552576130266290611', '-82051199267410638', '5785384030958665705', '-6133009717802725861', '-7871845376555109716', '4685212989500661509', '769223825795326523', '2104670239342855493', '-8999574577662585246', '-2848699779862371577', '4411672003276898792', '4175143834811808644', '7428902671467301472', '5921272806500930508', '-3836308983050930061', '86680290216010549', '-7443828569924995904', '4965206446783753340', '-6593786172448058094', '6567256020694440783', '4074207857477826513', '7670239873152567326', '497338908558145328', '-6542552796605339761', '2314832084165094217', '-3525227203262034027', '2487088749041530171', '9159474945745227690', '9222896082688958556', '-6137096914640534688', '5313339127491639524', '6149532469157898073', '6476987686131638681', '1326178481868679533', '-6210591518727488034', '3637112361529796791', '-4798644940681251731', '-3773136998192894793', '-5435822709474983888', '4117310771229415953', '-3207122683535425300', '-391362842439569968', '-5519810344344719209', '-4648190977869014937', '-7798268403935418913', '5901044345604054522', '5016535486339220484', '-2751055695362436606', '6166970216887978359', '8924269644784033060', '1666050318736529392', '-5949783439292862722', '-4989821726385625117', '3456316294813584769', '3749394332280061531', '-3557268865091350602', '-3483022574776694489', '-613469882960644450', '8207458569785267153', '4657503226949391435', '-8462321211398710610', '8239196429346800702', '-8829984544160063797', '7434921455110723746', '-7364071520792776728', '2775840251067771416', '-6267245747067997651', '6910034875550241881', '1590194633735686902', '2370235941034770516', '6941809078619452991', '3942183868284334157', '-2830385203718681412', '1286774360003318588', '7009417532076439325', '6224023053886040749', '4565031430335420073', '-3071864791753446059', '-4767536963343233422', '-6722600341974192522', '5769591337214981463', '4071490047770102600', '-949712490000915278', '7682837152639433138', '42487324306043315', '-7504840561121961744', '2683399827621403496', '-6065883316743076082', '3330919190414458241', '1416813601766951820', '3532370092125718013', '-5398089367086867949', '-2619341853819375368', '4308761492491441297', '-8092030669569316451', '-8677713436758202149', '1125226722819529917', '3429146202338791083', '-412704402378132556', '-2505837704901751715', '-2839506416532928017', '-722400225601668060', '2632634890820429404', '3822674477738969189', '-1175074723775913296', '1592153645701322107', '1059762400340964817', '922233723732129849', '-6750669421891187045', '-3792214777998468542', '439985319949089506', '-671343286763224953', '46933305726586862', '-652117253746563674', '6896156185069964276', '-4288683236394714411', '-686965618063115361', '-6735760016212039736', '-9035534225130269976', '641868616797443860', '7098599560398650583', '-8102463079154339600', '-1250790767300321061', '531920013532529375', '-976765514528998133', '-8850658511118646149', '-1883507824814220497', '-7744648380212957788', '-6689717885343199417', '1361489657346379965', '223889013848482079', '7976100939875308597', '-4746167706563604626', '399565880188880579', '-8324199546517189599', '388463458252091846', '2634309541804933826', '-23537822290095785', '-6066370770682792765', '-4806319896032592325', '7323463382570261703', '-7278046984885687081', '4912813542099489757', '4030650026377458651', '-7506324997539485270', '-8196578579339054694', '-9063281357515829584', '-1564037049376314780', '-6603844747009996534', '-6268305864372537722', '5652282845782833581', '3439377836653572305', '2449615471253129371', '-726932067100199690', '7722055373064048929', '-2362474540521017531', '1693597145337758027', '282488718934829285', '-8617840349864871396', '1818657866526910359', '-5978097463654951739', '6891267164470677438', '207124429295114270', '8367983021926652816', '-7580641375690397855', '-762930927176712682', '7094353521557135535', '8071139407818088476', '-2117335408198587344', '-4065577178194066983', '-436106574654356714', '4254051040099182295', '-5651879883851203954', '2551032459638156719', '356679829740305253', '-4676574229009765699', '-6218384608200546469', '571452456968526218', '706436158507269356', '-8235816179603269389', '-6489109777570502785', '-501538841033618609', '-3238535147717376429', '-6725782269108524479', '-5154798648094404240', '9076460978711502873', '2265055189082030896', '-3334913234103339998', '1469809628155902968', '2902315055353750734', '-8763004090664727773', '1215179917737773555', '-1601104511413497567', '3036700087778896814', '3915783006057137863', '-4340511208164380792', '4474533684829993472', '4581301801178681431', '2850339324882439414', '8455109717616396298', '1238109232605090078', '5897753078197024239', '-5424494704776884162', '631029820982838132', '-3924272032130150483', '2140438374364897', '-4149513062345101822', '6967621407255586589', '3303450059650204168', '-4595522361866648343', '-550870475070866888']
    #
    # dataset = [item for item in dataset if item["id"].replace(f"-{item['lang']}", "") in qids and item['lang'] not in ['ar', 'bn', 'en', 'fi', 'ja', 'ko', 'ru', 'te', 'pl', 'tr']]

    if "mkqa" in args.data_file:
        dataset = [item for item in dataset if item['lang'] not in ['ar', 'bn', 'en', 'fi', 'ja', 'ko', 'ru', 'te']]

        # questions = []
        # lang = "en"
        # with open(f'data/nq-dpr/nq-th-translated-query.jsonl') as f:
        #     for jsonline in f.readlines():
        #         qid, en_query, target_query, en_answer, target_answer, doc = json.loads(jsonline)
        #         questions.append(en_query.lower())
        # qids = set()
        # with open('./data/XOR-Full/mkqa_dev_full_original.jsonl') as f:
        #     for jsonline in f.readlines():
        #         example = json.loads(jsonline)
        #         if example['lang'] == lang:
        #             if example['question'].lower() in questions:
        #                 qids.add(example['id'].replace(f"-{lang}", ""))
        # dataset = [item for item in dataset if item["id"].replace(f"-th", "") not in qids]
        # print(len(dataset))

    if args.txt_file is True:
        tmp_preds = open(args.pred_file).read().split("\n")
        predictions = {}
        for item, pred in zip(dataset, tmp_preds):
            predictions[item["id"]] = pred
    else:
        if "*" in args.pred_file:
            predictions = {}
            for pred_file in glob.glob(args.pred_file):
                with open(pred_file) as prediction_file:
                    predictions.update(json.load(prediction_file))
        else:
            with open(args.pred_file) as prediction_file:
                predictions = json.load(prediction_file)

    if args.significant:
        outputs = calculate_f1_em_bleu_significant(dataset, predictions)
        with open("significant.txt", "w") as f:
            for q_id, f1 in outputs:
                f.write("{0}\t{1}\n".format(q_id, f1))
    else:
        results = calculate_f1_em_bleu(dataset, predictions)

        f1_total, em_total, bleu_total = 0.0, 0.0, 0.0
        total_num = 0
        lang_count = 0
        for lang in results:
            if results[lang]["count"] == 0:
                continue
            lang_count += 1
            f1_total += results[lang]["f1"]
            em_total += results[lang]["em"]
            bleu_total += results[lang]["bleu"]
            total_num += results[lang]["count"]
            print("Evaluating the performance on {0} for {1} examples".format(
                lang, results[lang]["count"]))
            print("F1: {0}, EM:{1}, BLEU:{2}".format(
                results[lang]["f1"] * 100, results[lang]["em"] * 100, results[lang]["bleu"] * 100))
        print("avg f1: {}".format(f1_total / lang_count * 100))
        print("avg em: {}".format(em_total / lang_count * 100))
        print("avg bleu: {}".format(bleu_total / lang_count * 100))


if __name__ == "__main__":
    main()