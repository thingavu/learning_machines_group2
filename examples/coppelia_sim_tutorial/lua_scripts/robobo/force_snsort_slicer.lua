local sim = require("sim")

function sysCall_init() end

function sysCall_trigger(inData)
    -- callback function automatically added for backward compatibility
    sim.breakForceSensor(inData.handle)
end
