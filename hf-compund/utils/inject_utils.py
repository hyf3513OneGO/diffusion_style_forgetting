def build_style_hooker():
    style_features=[]
    def style_hooker(module, fea_in, fea_out):
        style_features.append(fea_out)
    return style_hooker,style_features

def build_content_hooker():
    content_features=[]
    def content_hooker(module,fea_in,fea_out):
        content_features.append(fea_out)
    return content_hooker,content_features

def hook_model(layers,hook,model):
    for i,layer in model.named_children():
        if int(i) in layers:
            layer.register_forward_hook(hook)