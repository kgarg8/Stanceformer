for hf in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bert --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=0.50l_0.25h --mask=[[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bert --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 # baseline
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bert --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=0.50l_0.50h --mask=[[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bert --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=0.50l_1.00h --mask=[[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bert --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=1.00l_0.25h --mask=[[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bert --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=1.00l_0.50h --mask=[[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bert --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf
done

for hf in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bertweet --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=0.50l_0.25h --mask=[[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bertweet --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 # baseline
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bertweet --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=0.50l_0.50h --mask=[[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bertweet --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=0.50l_1.00h --mask=[[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bertweet --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=1.00l_0.25h --mask=[[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bertweet --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf --mask_str=1.00l_0.50h --mask=[[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]]
    CUDA_VISIBLE_DEVICES=0 python train_automodel.py --lr=5e-5 --model=Bertweet --dataset=SemEval15 --aspect=restaurants --gs_sheet=ABSA_py_linked3 --highlight --highlight_factor=$hf
done