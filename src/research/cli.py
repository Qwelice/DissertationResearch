import typer


app = typer.Typer()


def main():
    app()


if __name__ == '__main__':
    from research.engine.lightning.setup import setup

    data, model, trainer = setup()
    trainer.fit(model=model,
                datamodule=data)

    # main()