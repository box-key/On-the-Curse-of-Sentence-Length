import torch
import time
import math

class TrainModel():

    def __init__(self, model, train_iterator, val_iterator, optimizer, criterion, model_type, weight_initializer=None):
        self.model = model
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_type = model_type
        if weight_initializer==None:
            self.weight_initializer = TrainModel.default_init_weight
        else:
            self.weight_initializer = weight_initializer
    
    def train(self, clip):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.train_iterator):
            if self.model_type == 'Attention':
                src, src_len = batch.src
            else:
                src = batch.src
            trg = batch.trg

            self.optimizer.zero_grad()
            if self.model_type == 'Transformer':
                output, _ = self.model(src, trg[:,:-1])
            elif self.model_type == 'Attention':
                output = self.model(src, src_len, trg)
            else:
                output = self.model(src, trg)
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]

            if self.model_type == 'Transformer':
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
            else:
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = self.criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_iterator)

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_iterator):
                if self.model_type == 'Attention':
                    src, src_len = batch.src
                else:
                    src = batch.src
                trg = batch.trg

                if self.model_type == 'Transformer':
                    output, _ = self.model(src, trg[:,:-1]) #turn off teacher forcing
                elif self.model_type == 'Attention':
                    output = self.model(src, src_len, trg, 0) #turn off teacher forcing
                else:
                    output = self.model(src, trg, 0) #turn off teacher forcing
                #trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]
                output_dim = output.shape[-1]

                if self.model_type=='Transformer':
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg[:,1:].contiguous().view(-1)
                else:
                    output = output[1:].view(-1, output_dim)
                    trg = trg[1:].view(-1)

                #trg = [(trg len - 1) * batch size]
                #output = [(trg len - 1) * batch size, output dim]
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(self.val_iterator)

    def epoch(self, n_epochs, clip, model_name='tut-model.pt'):
        # Initialize weights
        self.model.apply(self.weight_initializer)
        # Keep track of the best model (the one with minimum validation loss)
        best_valid_loss = float('inf')
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss = self.train(clip)
            valid_loss = self.evaluate()
            epoch_mins, epoch_secs = TrainModel.epoch_time(start_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), model_name)
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    @staticmethod
    def epoch_time(start_time):
        total = time.time() - start_time
        return int(total/60), int(total%60)
    
    @staticmethod
    def default_init_weights(model):
        for name, param in model.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.01)
