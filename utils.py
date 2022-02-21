import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_conversation(text,tokenizer,model):
    input_ids = tokenizer.encode(
        text + '</s>', 
        return_tensors='pt', 
        padding=True, 
        truncation=True,
        max_length = 512).to(device)
    with torch.no_grad():
        output = model.to(device).generate(input_ids=input_ids, max_length=3)
    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0]
    return label

def freeze_params(model):
    for par in model.parameters():
        par.requires_grad = False