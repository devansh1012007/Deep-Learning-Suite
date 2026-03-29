gradcam = GradCAM(model, target_layer=model.layer4)

heatmap = gradcam.generate(x, class_idx=3)