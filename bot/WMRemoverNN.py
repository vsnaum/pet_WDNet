from PIL import Image
import onnxruntime
import numpy as np



def resize_for_kernel(pil_img,kernel_size=16):
    src_img_size = pil_img.size
    new_w = (src_img_size[0]//kernel_size + 1)*kernel_size if src_img_size[0]%kernel_size != 0 else src_img_size[0]
    new_h = (src_img_size[1]//kernel_size + 1)*kernel_size if src_img_size[1]%kernel_size != 0 else src_img_size[1]
    if (new_w != src_img_size[0]) or (new_h != src_img_size[1]):
        return (pil_img.resize((new_w,new_h)), src_img_size)
    else:
        return (pil_img, src_img_size)

def proc_with_nn(onnx_model_path,pil_img):
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    np_img = np.array(pil_img) / 255
    np_img = np_img.astype(np.float32)
    np_img = np_img[np.newaxis,:,:,:]
    np_img = np.transpose(np_img,[0,3,1,2])
    ort_inputs = {ort_session.get_inputs()[0].name: np_img}
    return ort_session.run(None, ort_inputs)

def run(onnx_model_path,pil_img):
    img, src_size = resize_for_kernel(pil_img)
    img_nowm = proc_with_nn(onnx_model_path,img)
    img_nowm = img_nowm[0].squeeze(0)
    img_nowm = np.transpose(img_nowm * 255,[1,2,0]).astype(np.uint8)
    return Image.fromarray(img_nowm).resize(src_size)