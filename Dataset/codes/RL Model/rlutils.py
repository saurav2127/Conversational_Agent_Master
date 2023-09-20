import numpy as np
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu as sentblu
from nltk.translate.meteor_score import meteor_score
import nltk
#nltk.download('wordnet')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import pdb
import spacy

def convert_sentences_to_strings(sentences:list, tokenizer):
    str_sentences = []
    for i in sentences:
        str_sentences.append(tokenizer.decode(i.tolist()[0][2:-2])) # Excludeqs the zero shot tokens: {A:, B:} and the End of turn tokens: [628, 198]
    return str_sentences

def normalize(text, nlp):
    sent = ''
    doc = nlp(text)
    for token in doc:
        if not token.is_punct:
            sent += token.lemma_
            sent += ' '
    return sent

def jaccard_similarity(context_sentence_list, generated_sentence, tokenizer, nlp):
    str1 = context_sentence_list[0]
    str1 = normalize(str1, nlp)
    str1 = set(str1.split())
    similarity_score = []
    jacc_score = []
    generated_sentences = convert_sentences_to_strings(generated_sentence, tokenizer)
    for i in generated_sentences:
        str2 = i
        str2 = normalize(str2, nlp)
        str2 = set(str2.split())
        sim_score = float(len(str1 & str2)) / len(str1 | str2)
        jacc_score.append(sim_score)
        if sim_score >= 0.5:
            similarity_score.append(-3)
        elif (sim_score < 0.5) and (sim_score > 0.2):
            similarity_score.append(-1)
        else:
            similarity_score.append(3)
    return similarity_score, jacc_score

def calculate_blue4(reference, candidates, tokenizer):
    reference = [reference.split()]
    if isinstance(candidates, list):
        blu = []
        for i in candidates:
            #print("Reference {} and \n candidate is {}".format(reference, tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split()))
            blu.append(sentblu(reference, tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split()))
        return blu
    else:
        return sentblu(reference, candidates.split())

def calculate_meteor(reference, candidates, tokenizer):
    meteor_scores = []
    scores = []
    for i in candidates:
        score = meteor_score([reference.split()], tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split())
        meteor_scores.append(score)
        if score >= 0.7:
            scores.append(3)
        elif (score >= 0.4) and (score < 0.7):
            scores.append(2)
        else:
            scores.append(-2)
    return scores, meteor_scores

def calculate_emotion_prob(generated_sentences, classifier, actual_emotion_label, tokenizer, device):
    gen_str_sent = convert_sentences_to_strings(generated_sentences, tokenizer)
    pred_labels, logits = classifier.predict(gen_str_sent)
    #probs = F.softmax(torch.tensor(logits), dim=-1)
    #probs = probs[:, actual_emotion_label]
    logits = logits[:, actual_emotion_label]
    return logits.tolist(), list(pred_labels)

def get_emotion_reward(generated_sentences, classifier, actual_emotion_label, tokenizer, beta):
    gen_str_sent = convert_sentences_to_strings(generated_sentences, tokenizer)
    pred_labels, logits = classifier.predict(gen_str_sent)
    probs = F.softmax(torch.tensor(logits), dim=-1)
    actual_label_prob = probs[:, actual_emotion_label]
    other_label_prob = probs.sum(-1) - actual_label_prob
    reward_dict = {'actual_prob': actual_label_prob.tolist(),
                   'other_prob': other_label_prob.tolist()}
    reward = actual_label_prob - beta * other_label_prob
    return reward_dict, reward.tolist()

def robert_emotion_reward(generated_sentences, classifier, actual_emotion_label, tokenizer, persuasion_tokenizer, beta, device):
    gen_str_sent = convert_sentences_to_strings(generated_sentences, tokenizer)
    inputs = persuasion_tokenizer(gen_str_sent, return_tensors='pt', padding=True)
    output = classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    probs = F.softmax(output.logits, dim=-1)
    actual_label_prob = probs[:, actual_emotion_label]
    other_label_prob = probs.sum(-1) - actual_label_prob
    reward_dict = {'actual_prob': actual_label_prob.tolist(),
                   'other_prob': other_label_prob.tolist()}
    reward = actual_label_prob - beta * other_label_prob
    return reward_dict, reward.tolist()

def get_persuasion_reward(generated_sentences, persuasion_classifier, actual_persuasion_label, device, tokenizer, persuasion_tokenizer, beta2):
    #pdb.set_trace()
    gen_str_sent = convert_sentences_to_strings(generated_sentences, tokenizer)
    inputs = persuasion_tokenizer(gen_str_sent, return_tensors='pt', padding=True, truncation=True)
    output = persuasion_classifier(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    probs = F.softmax(output.logits, dim=-1)
    actual_label_prob = probs[:, actual_persuasion_label]
    other_label_prob = probs.sum(-1) - actual_label_prob
    reward_dict = {'actual_prob': actual_label_prob.tolist(),
                   'other_prob': other_label_prob.tolist()}
    reward = actual_label_prob - beta2 * other_label_prob
    return reward_dict, reward.tolist()

def calculate_rewards(current_sentence, generated_sentences, source_list, tokenizer, use_jacc, use_meteor, nlp, device, beta, beta2, persuasion_tokenizer,
                      gamma1, gamma2, gamma3, gamma4, persuasion_classifier, actual_persuasion_label, bleu=False, actual_emotion_label=None, classifier=None,
                      persuader=False):
    scores = {}
    scores['meteor'] = []
    scores['jacc'] = []
    scores['emotion'] = []
    scores['persuasion'] = []
    scores['emotion_actual_prob'] = []
    scores['emotion_other_prob'] = []
    scores['persuasion_actual_prob'] = []
    scores['persuasion_other_prob'] = []

    if len(generated_sentences) >= 1:
        rewards = np.zeros((len(generated_sentences)))
        if (len(source_list) ==2):
            if use_jacc: # use jaccard similarity
                similarity_score, act_similarity_score = jaccard_similarity(source_list, generated_sentences, tokenizer, nlp)
                rewards -= gamma1*np.array(act_similarity_score)
            else: # use cosine similarity
                cosine_similarity = calculate_cosine_similarity()
        if bleu:
            blu_score = calculate_blue4(current_sentence, generated_sentences, tokenizer)
            rewards += np.array(blu_score)
        if use_meteor:
            #act_meteor_score = []
            #meteor_score =[]
            meteorscore, act_meteor_score = calculate_meteor(current_sentence, generated_sentences, tokenizer)
            rewards += gamma2*np.array(act_meteor_score)
        if not use_meteor:
            meteorscore = None
            act_meteor_score = None
        if classifier:
            #probs, pred_label = calculate_emotion_prob(generated_sentences, classifier, actual_emotion_label, tokenizer, device)
            #emotion_dict, emotion_probs = get_emotion_reward(generated_sentences, classifier, actual_emotion_label, tokenizer, beta)
            emotion_dict, emotion_probs = robert_emotion_reward(generated_sentences, classifier, actual_emotion_label, tokenizer, persuasion_tokenizer, beta, device)
            rewards += gamma3*np.array(emotion_probs)
        if not classifier:
            emotion_dict = {'actual_prob': [None], 'other_prob': [None]}
            emotion_probs = [None]
        if persuasion_classifier:   # generated_sentences, persuasion_classifier, actual_persuasion_label, device, tokenizer, persuasion_tokenizer, beta2
            persuasion_dict, persuasion_probs = get_persuasion_reward(generated_sentences, persuasion_classifier, actual_persuasion_label,
                                                                      device, tokenizer, persuasion_tokenizer, beta2)
            rewards += gamma4* np.array(persuasion_probs)
        if not persuasion_classifier:
            persuasion_dict =  {'actual_prob': [None], 'other_prob': [None]}
            persuasion_probs = [None]
    else:
        rewards = 0
        similarity_score, act_similarity_score = jaccard_similarity(current_sentence, generated_sentences, tokenizer, nlp)
        if similarity_score >= 0.5:
            rewards -= similarity_score
        blue_score = calculate_blue4(current_sentence, generated_sentences, tokenizer)
        rewards += blue_score
    try:
        scores['jacc'].extend(act_similarity_score)
    except:
        pass
    
    scores['meteor'].extend(act_meteor_score)
    scores['emotion'].extend(emotion_probs)
    scores['persuasion'].extend(persuasion_probs)
    scores['emotion_actual_prob'].extend(emotion_dict['actual_prob'])
    scores['emotion_other_prob'].extend(emotion_dict['other_prob'])
    scores['persuasion_actual_prob'].extend(persuasion_dict['actual_prob'])
    scores['persuasion_other_prob'].extend(persuasion_dict['other_prob'])

    return list(rewards), scores

def append(generated_list, context_sentence, tokenizer):
    if len(generated_list) == 2:
        generated_list.pop(0)
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    else:
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    return generated_list

def expand_inputs_for_N_candidates(inputs, num_candidates):
    # inputs = inputs[None, ...]
    return inputs.repeat((num_candidates, 1))

def modify_generated_sequence(generated_sequences, generated_log_probs):
    final_generated_sequences = []
    final_generated_log_probs = []
    for i in range(generated_sequences.shape[0]):
        batch_tokens = []
        batch_log_probs = []
        for j in range(len(generated_sequences[i])):
            if generated_sequences[i][j] != 628 and generated_sequences[i][j] != -1:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
            elif generated_sequences[i][j] == 628:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
                batch_tokens.append(198)
                break
            else:
                break
        final_generated_sequences.append(torch.tensor(batch_tokens).unsqueeze(0))
        ### BE CAREFUL WHEN USING THIS, SINCE IT DOESN NOT AVERAGES THE LOG PROBS INSTEAD
        ### IT JUST TAKES THE SUM.
        final_generated_log_probs.append(torch.tensor(batch_log_probs).sum().item())
    return final_generated_sequences, final_generated_log_probs

def top_p_candidates(logits, prob=0.92, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cum_sum > prob
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
    #indices_to_remove = sorted_indices_to_remove.scatter(1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits

def generate_n_candidates(model, inputs, top_p,  temperature, num_candidates, max_gen_length, past,
                          device, eos_token_id=628, pad_token_id=198):
    curr_len = 2
    inputs = expand_inputs_for_N_candidates(inputs, num_candidates)
    inputs_ = inputs
    generated_sequences = torch.ones((inputs.shape[0], max_gen_length), dtype=torch.long) * -1
    generated_sequences[:, 0:2] = inputs.cpu()
    generated_token_log_prob = torch.zeros((inputs.shape[0], max_gen_length), dtype=torch.float)
    unfinished_sequences = inputs.new(inputs.shape[0]).fill_(1) #.cpu()
    i = 0
    while True:
        if past:
            if past[0][0].shape[-2] > 1024:
                if not torch.all(generated_sequences==-1):
                    final_generated_sequence, final_generated_log_probs = modify_generated_sequence(generated_sequences, generated_token_log_prob)
                    return final_generated_sequence, final_generated_log_probs, past_to_return
                else:
                    return None, None
        outputs = model(inputs, past, return_dict=False)
        logits, past = outputs[0], outputs[1]
        next_token_logits = logits[:, -1, :].contiguous() / temperature
        if top_p and top_p > 0.0:
            # This returns score after performing softmax function.
            next_token_logits = top_p_candidates(next_token_logits, top_p)
            next_token_log_probs = F.log_softmax(next_token_logits, -1)
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
            next_tokens = next_tokens.squeeze(1)
            if eos_token_id is not None:
                assert pad_token_id is not None # "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            # NOTE: SAVE LOG PROBS AS WELL
            generated_sequences[:, curr_len] = next_tokens.cpu()
            inputs = next_tokens.unsqueeze(1).to(device)
            #inputs_ = torch.cat((inputs_, next_tokens[:, None]), dim=-1)
            curr_len = curr_len + 1
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break
            if curr_len >= max_gen_length:
                break
    final_generated_sequences, final_generated_log_probs =  modify_generated_sequence(generated_sequences, generated_token_log_prob)
    return final_generated_sequences, final_generated_log_probs

def compute_log_probs(target_token_ids, logits, mask, average_sent_loss=False):
    logits = logits[:, :-1, :].contiguous() # (batch, sequence_length, vocab_size)
    target_token_ids = target_token_ids[:, 1:].contiguous() # (batch, sequence_length)
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, target_token_ids.unsqueeze(-1)).squeeze(-1)
    mask = mask[:, 1:].contiguous()
    if average_sent_loss:
        log_probs = (log_probs * mask).sum(-1) / mask.sum(-1)
    else:
        log_probs = (log_probs * mask).sum(-1)
    return {'log_probs': log_probs}

def ppo_step(model_A, model_B, buffer_memory, device, beta, ppo_epsilon, num_candidates, optimizer, dial_inputs,
             role_ids, train_single_model=False, model_to_train=None, average_sent_loss=False, use_recent_past=False):

    optimizer.zero_grad()
    log_dict = {}
    new_log_prob = []
    old_log_prob = []
    rewardlist = []
    ratios = []
    policy_loss = []
    advantages  = []

    if use_recent_past:
        print('USING RECENT PAST')
    else:
        print('NOT USING RECENT PAST')

    '''if use_recent_past:
        batches = buffer_memory.get_batch(shuffle=False)
        past = None
        for idx, batch in enumerate(batches):
            action = torch.tensor(batch['action'], device=device).unsqueeze(0)

            old_log_probs = torch.tensor(batch['log_prob'], device=device).unsqueeze(0)
            old_log_prob.append(batch['log_prob'])

            rewards = torch.tensor(batch['reward'], device=device).unsqueeze(0)
            rewardlist.append(batch['reward'])

            state = torch.tensor(batch['state'], device=device).unsqueeze(0)
            if batch['human_response']:
                if idx == 0:
                    logits, _ = model_A(torch.tensor(batch['state'], device=device).unsqueeze(0), past, return_dict=False)
                    new_log_probs = compute_log_probs(target_token_ids=action,
                                                      logits=logits,
                                                      mask=torch.ones_like(action).to(device),
                                                      average_sent_loss=average_sent_loss)['log_probs']
                if idx > 0 and idx % (num_candidates + 1) == 0: # {6, 12, 18, ...} // 5 (num_cand) + 1
                    history_indices = idx // (num_candidates + 1)
                    history = dial_inputs[:history_indices*2]
                    ## NOTE: ALSO TRY COLLECTING PASS FROM MODEL_B AND MODEL_A RECURSIVELY.
                    past = get_past(history, model_A, device)
                    #past = get_recursive_past(history, role_ids[:history_indices*2], model_A, model_B, device)
                    logits, _ = model_A(action, past_key_values=past, return_dict=False)
                    new_log_probs = compute_log_probs(target_token_ids=action,
                                                      logits=logits,
                                                      mask=torch.ones_like(action).to(device),
                                                      average_sent_loss=average_sent_loss)['log_probs']
            else:
                ## NOTE: In order to calculate past for generated candidates we do not need
                ##       current utterance, but just the previous utterances.
                history_indices = idx // (num_candidates + 1)  # {(1,2,3,4,5), (7,8,9,10,11), (13,14,15,16,17)}
                history = dial_inputs[:history_indices*2]
                if len(history) != 0:
                    past = get_past(history, model_A, device)
                    #past = get_recursive_past(history, role_ids[:history_indices*2], model_A, model_B, device)
                else:
                    past = None
                logits, _ = model_A(action, past_key_values=past, return_dict=False)
                new_log_probs = compute_log_probs(target_token_ids=action,
                                                  logits=logits,
                                                  mask=torch.ones_like(action).to(device),
                                                  average_sent_loss=average_sent_loss)['log_probs']
            advantages = rewards
            new_log_prob.append(new_log_probs.item())
            ratio = (new_log_probs - old_log_probs).exp()
            ratios.append(ratio.item())
            policyloss1 = - advantages * ratio
            policyloss2 = - advantages * ratio.clamp(1 - ppo_epsilon, 1 + ppo_epsilon)
            policyloss = torch.min(policyloss1, policyloss2).mean()
            policyloss /= len(batches)
            policyloss.backward()
            policy_loss.append(policyloss.item())

            del past
            del action
            del logits
            del rewards
            del new_log_probs
            del old_log_probs
            torch.device.empty_cache()
            
        with torch.no_grad():
            log_dict['policy_loss'] = np.mean(policy_loss)
            print('Policy Loss: ', log_dict['policy_loss'])
            # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
            log_dict['approx_kl'] = np.mean((np.exp(np.array(new_log_prob) - np.array(old_log_prob)) - 1)\
                                            - (np.array(new_log_prob) - np.array(old_log_prob)))
            #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
            print('approx KL div: ', log_dict['approx_kl'])
            log_dict['clip_frac'] = (np.abs(np.array(ratios)-1) > ppo_epsilon).mean()
            print('clip frac: ', log_dict['clip_frac'])
            log_dict['reward'] = np.mean(rewardlist)
            print('rewards: ', log_dict['reward'])'''

    if use_recent_past:
        batches = buffer_memory.get_batch(shuffle=False)
        past = None
        i = 1
        for idx, batch in enumerate(batches):
            action = torch.tensor(batch['action'], device=device).unsqueeze(0)
            if batch['human_response']:
                if idx == 0:
                    logits, past = model_A(action, past, return_dict=False)
                if idx > 0 and idx % (num_candidates + 1) == 0:
                    try:
                        past = out
                    except:
                        pass
                    #history_indices = idx // (num_candidates + 1)
                    #history = dial_inputs[history_indices]
                    history = dial_inputs[i]
                    _, past = model_A(history.to(device), past_key_values=past, return_dict=False)
                    logits, out = model_A(action, past_key_values=past, return_dict=False)
                    i += 2
            else:
                history_indices = idx // (num_candidates + 1)  # {A:(1,2,3,4,5),B, C:(7,8,9,10,11), D, E: (13,14,15,16,17)}
                if history_indices == 0:
                    logits, _ = model_A(action, past_key_values=None, return_dict=False)
                else:
                    logits, _ = model_A(action, past_key_values=past, return_dict=False)
            
            new_log_probs = compute_log_probs(target_token_ids=action,
                                              logits=logits,
                                              mask=torch.ones_like(action).to(device),
                                              average_sent_loss=average_sent_loss)['log_probs']

            old_log_probs = torch.tensor(batch['log_prob'], device=device).unsqueeze(0)
            old_log_prob.append(old_log_probs)

            rewards = torch.tensor(batch['reward'], device=device).unsqueeze(0)
            rewardlist.append(batch['reward'])
            advantages.append(rewards)

            new_log_prob.append(new_log_probs)

        new_log_prob = torch.cat(new_log_prob, dim=-1)
        old_log_prob = torch.cat(old_log_prob, dim=-1)
        advantages = torch.cat(advantages, dim=-1)
        ratio = (new_log_prob - old_log_prob).exp()
        policyloss1 = - advantages * ratio
        policyloss2 = - advantages * ratio.clamp(1 - ppo_epsilon, 1 + ppo_epsilon)
        policyloss = torch.min(policyloss1, policyloss2).mean()
        policyloss.backward()

        with torch.no_grad():
            log_dict['policy_loss'] = policyloss.item()
            print('Policy Loss: ', log_dict['policy_loss'])
            # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
            log_dict['approx_kl'] = torch.mean(((new_log_prob - old_log_prob).exp() - 1)\
                                            - (new_log_prob - old_log_prob)).item()
            #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
            print('approx KL div: ', log_dict['approx_kl'])
            log_dict['clip_frac'] = torch.mean((torch.abs(ratio-1) > ppo_epsilon).float()).item()
            print('clip frac: ', log_dict['clip_frac'])
            log_dict['reward'] = np.mean(rewardlist)
            print('rewards: ', log_dict['reward'])

    if not train_single_model:
        nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
    else:
        if model_to_train =='persuader':
            nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)

    optimizer.step()
    return log_dict

@torch.no_grad()
def collect_samples(batch, model_A, model_B, top_p, eos_token_id, pad_token_id, max_gen_length, num_candidates, human_reward, use_bleu, use_meteor, # if not bleu meteor is automatically used.
                    use_jacc, buffer_memory, device, tokenizer, classifier, temperature, use_recent_past, average_sent_loss, nlp, beta, beta2,  persuasion_classifier,
                    gamma1, gamma2, gamma3, gamma4, persuasion_tokenizer, train_single_model=False, model_to_train=None, recompute_log_prob=True, fp16=False):

    scores_dict = {}
    scores_dict['meteor_scores'] = []
    scores_dict['jacc_scores'] = []
    scores_dict['emotion_scores'] = []
    scores_dict['emotion_actual_prob'] = []
    scores_dict['emotion_other_prob'] = []
    scores_dict['persuasion_actual_prob'] = []
    scores_dict['persuasion_other_prob'] = []
    scores_dict['persuasion_scores'] = []

    #candidate_dict = {}


    if not persuasion_classifier and classifier:
        role_ids, dialog_tokens, emotion_labels = batch
    elif persuasion_classifier and classifier:
        role_ids, dialog_tokens, emotion_labels, persuasion_label = batch
    elif persuasion_classifier and not classifier:
        role_ids, dialog_tokens, persuasion_label = batch
    else:
        role_ids, dialog_tokens = batch
    
    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]

    past = None
    past_ = None
    context = None
    cntxt = None
    persuader_generated_list, persuadee_generated_list = [], []

    for num_turn, dialog_turn_inputs in enumerate(dial_inputs):
        assert not np.any(np.isnan(dialog_turn_inputs).cpu().numpy()), 'Inputs Dialog contains Nan value.'
        dialog_turn_inputs = dialog_turn_inputs.to(device)

        if classifier and not persuasion_classifier:
            actual_emotion_label = emotion_labels[num_turn]
            actual_persuasion_label = None
        elif classifier and  persuasion_classifier:
            actual_emotion_label = emotion_labels[num_turn]
            actual_persuasion_label = persuasion_label[num_turn]
        elif persuasion_classifier and not classifier:
            actual_emotion_label = None
            actual_persuasion_label = persuasion_label[num_turn]
        else:
            actual_emotion_label = None
            actual_persuasion_label = None

        if model_to_train == 'persuader':
            if role_ids[num_turn] == 0:
                '''if use_recent_past:
                    if cntxt is not None:
                        past = prepare_inputs(cntxt, model_A)
                    else:
                        past = None'''
                
                #dial_turn_str = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]

                outputs = model_A(dialog_turn_inputs, past, return_dict=False)
                logits = outputs[0]

                mask = torch.ones_like(dialog_turn_inputs).to(device)
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs, logits=logits, mask=mask, average_sent_loss=average_sent_loss)
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            persuader=True,
                                            human_response=True)
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_A,
                                                                                     torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device),
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    output = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
                    past_ = output[1]
                else:
                    '''Here first we calculate the past based on the context sentence and then we generate candidates.'''
                    '''if cntxt is not None:
                        past_ = prepare_inputs(expand_inputs_for_N_candidates(cntxt, num_candidates), model_A)
                    else:
                        past_ = None'''

                    generated_sequence, generated_log_probs = generate_n_candidates(model_A,
                                                                                   torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device), top_p,
                                                                                   eos_token_id=eos_token_id,
                                                                                   pad_token_id=pad_token_id,
                                                                                   num_candidates=num_candidates,
                                                                                   max_gen_length=max_gen_length,
                                                                                   temperature=temperature,
                                                                                   past=past_,
                                                                                   device=device)

                #gen_sent = convert_sentences_to_strings(generated_sequence, tokenizer)
                persuader_generated_list = append(persuader_generated_list, dialog_turn_inputs, tokenizer)
                reward, scores = calculate_rewards(current_sentence=tokenizer.decode(dialog_turn_inputs.tolist()[0][2:]).split('\n')[0],
                                           generated_sentences= generated_sequence,
                                           actual_emotion_label=actual_emotion_label,
                                           actual_persuasion_label=actual_persuasion_label,
                                           source_list=persuader_generated_list,
                                           tokenizer=tokenizer,
                                           persuasion_tokenizer=persuasion_tokenizer,
                                           classifier=classifier,
                                           persuasion_classifier=persuasion_classifier,
                                           persuader=True,
                                           use_jacc=use_jacc,
                                           bleu=use_bleu,
                                           use_meteor=use_meteor,
                                           nlp=nlp,
                                           device=device,
                                           beta=beta,
                                           beta2=beta2, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, gamma4=gamma4)

                #candidate_dict[dial_turn_str] = convert_sentences_to_strings(generated_sequence, tokenizer)

                scores_dict['meteor_scores'].extend(scores['meteor'])
                scores_dict['jacc_scores'].extend(scores['jacc'])
                scores_dict['emotion_scores'].extend(scores['emotion'])
                scores_dict['emotion_actual_prob'].extend(scores['emotion_actual_prob'])
                scores_dict['emotion_other_prob'].extend(scores['emotion_other_prob'])
                scores_dict['persuasion_actual_prob'].extend(scores['persuasion_actual_prob'])
                scores_dict['persuasion_other_prob'].extend(scores['persuasion_other_prob'])
                scores_dict['persuasion_scores'].extend(scores['persuasion'])

                if recompute_log_prob:
                    for i in range(len(generated_sequence)):
                        # NOTE: STILL USING THE PAST FROM PREVIOUS UTTERANCE, SINCE WE DO NOT NEED PAST FROM
                        #       CONTAINING CURRENT UTTERANCE for GENERATED CANDIDATES
                        output = model_A(generated_sequence[i].to(device), past_key_values=past, return_dict=False)
                        logits = output[0]
                        log_probs = compute_log_probs(target_token_ids=generated_sequence[i].to(device),
                                                      logits=logits,
                                                      mask=torch.ones_like(generated_sequence[i]).to(device),
                                                      average_sent_loss=average_sent_loss)['log_probs'].item()
                        buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                                    context=context,
                                                    action= generated_sequence[i].tolist()[0],
                                                    action_log_probs=log_probs,
                                                    reward=reward[i],
                                                    persuader=True,
                                                    human_response=False)
                else:
                    for i in range(len(generated_sequence)):
                        buffer_memory.update_buffer(state=dialog_turn_inputs.tolis()[0],
                                                    action=generated_sequence[i].tolist()[0],
                                                    action_log_probs=generated_log_probs[i],
                                                    persuader=True,
                                                    human_response=False)
                past = outputs[1]
                outputs = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
                past_ = outputs[1]
            else:
                #NOTE: Context will always be persuadee's utterance since, because candidates are generated in response to this utterance.
                outputs = model_A(dialog_turn_inputs, past, return_dict=False)
                past = outputs[1]
                outputs = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
                past_ = outputs[1]
        context = dialog_turn_inputs.tolist()[0]
        cntxt = dialog_turn_inputs

    return dial_inputs, role_ids, scores_dict, #candidate_dict

def get_past(batches, model, device):
    states = torch.cat(batches, dim=-1).to(device)
    outputs = model(states, past_key_values=None, return_dict=False)
    return outputs[1]

def prepare_inputs_for_model(batches, model, num_candidates, device):
    states = get_history_utterances(batches, num_candidates)
    states = torch.cat(states, dim=1, device=device)
    outputs = model(states, past_key_values=None, return_dict=False)
    return outputs[1]

def get_history_utterances(batches, num_candidates):
    states = []
    for i in range(0, len(batches), num_candidates+1):
        states.append(i)
    return states

def get_recursive_past(dial_inputs, role_ids, model_A, model_B, device):
    '''
    Uses both models alternatively to calculate pasts.
    Used in case of training only the persuader.
    '''
    past = None
    for num_turn, utter in enumerate(dial_inputs):
        if role_ids[num_turn] == 0:
            _, past = model_A(utter.to(device), past_key_values=past, return_dict=False)
        else:
            _, past = model_B(utter.to(device), past_key_values=past, return_dict=False)
    return past
