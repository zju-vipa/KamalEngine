import torch.optim as optim
import torch.nn as nn
import torch

def eval(model, criterion, test_loader, metric, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric.reset()
    model.to(device)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img, target = img.to(device), target.to(device)
            out = model(img)
            pred = out.max(1)[1]
            loss = criterion( out, target )
            val_loss+=loss.detach().cpu().numpy()
            metric.update( pred, target)
    return metric.get_results(return_key_metric=True), val_loss/len(test_loader)

def train(model, criterion, optimizer, scheduler, train_loader, 
          test_loader, metric, val_criterion=None, pth_path=None, 
          total_epochs=30, total_itrs=None, val_interval=None, verbose=False, weights_only=True):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_score = -1
    best_val_loss = 999999

    cur_itr = 1
    cur_epoch = 1

    if total_itrs is None:
        total_itrs = len(train_loader)*total_epochs
    else:
        total_epochs = total_itrs // len(train_loader)

    if val_interval is None:
        val_interval = len(train_loader)

    if val_criterion is None:
        val_criterion = criterion

    while True:
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            if cur_itr%10==0 and verbose:
                print("Epoch %d/%d, Batch %d/%d, iter %d/%d, loss=%.4f"%(cur_epoch, total_epochs, i+1, len(train_loader), cur_itr, total_itrs, loss.item()))

            if cur_itr%val_interval==0:
                model.eval()
                (metric_name, score), val_loss = eval(model=model, 
                                                       criterion=val_criterion, 
                                                       test_loader=test_loader, 
                                                       metric=metric, 
                                                       device=device)
                print("[TEST] Epoch %d/%d, val_loss=%.4f, %s=%.4f\n"%(cur_epoch, total_epochs, val_loss, metric_name, score))

                if best_score<score:
                    if pth_path is not None:
                        if weights_only:
                            torch.save( model.state_dict(), pth_path )
                        else:
                            torch.save( model, pth_path )
                    best_score=score
                    best_val_loss=val_loss
                model.train()

            if scheduler is not None:
                scheduler.step()
            
            if cur_itr==total_itrs:
                print("val_loss=%.4f, best %s=%.4f"%(best_val_loss, metric_name, best_score))
                return best_score, best_val_loss
            cur_itr+=1
        cur_epoch+=1

def kd(student, teacher, criterion, optimizer, scheduler, train_loader, 
          test_loader, metrics, val_criterion=None, pth_path=None, total_epochs=30, total_itrs=None, val_interval=None, verbose=False):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student.train().to(device)
    teacher.eval().to(device)

    best_score = -1
    best_val_loss = 999999
    cur_itr = 1
    cur_epoch = 1

    if total_itrs is None:
        total_itrs = len(train_loader)*total_epochs
    else:
        total_epochs = total_itrs // len(train_loader)
    if val_interval is None:
        val_interval = len(train_loader)
    if val_criterion is None:
        val_criterion = criterion

    while True:
        student.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            s_out = student(img)
            with torch.no_grad():
                t_out = teacher(img)
            loss = criterion(s_out, t_out)
            loss.backward()
            optimizer.step()

            if cur_itr%10==0 and verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(cur_epoch, total_epochs, cur_itr, len(train_loader), loss.item()))
            
            if cur_itr%val_interval==0:
                student.eval()
                (metric_name, score), val_loss = eval(model=student,
                                                        criterion=nn.CrossEntropyLoss(), 
                                                        test_loader=test_loader, 
                                                        metrics=val_criterion, 
                                                        device=device)
                print("Epoch %d/%d, iter %d/%d val_loss=%.4f, %s=%.4f\n"%(cur_epoch, total_epochs, cur_itr, total_itrs, val_loss, metric_name, score))
                if best_score<score:
                    if pth_path is not None:
                        torch.save( student, pth_path )
                        print("Best model saved as %s"%(pth_path))
                    best_score=score
                    best_val_loss=val_loss
                student.train()
            if cur_itr == total_itrs:
                print("val_loss=%.4f, best %s=%.4f"%(best_val_loss, metric_name, best_score))
                return best_score, best_val_loss
            cur_itr+=1
            if scheduler is not None:
                scheduler.step()
        cur_epoch+=1
