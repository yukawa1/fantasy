from . import main

# The guard matters: parallel fits start worker processes that re-import the
# main module (macOS/Windows "spawn"); without it each would start a server.
if __name__ == "__main__":
    main()
