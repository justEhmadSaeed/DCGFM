from gp.lightning.module_template import IBBaseTemplate

class IBGraphPredLightning(IBBaseTemplate):
    def forward(self, batch):
        return self.model(batch)