import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
    {
        fullName: {
            type: String,
            required: true,
        },
        phoneNumber: {
            type: String,
            required: true,
        },
        email: {
            type: String,
            required: true,
            unique: true,
            lowercase: true,
        },
        passwordHash: {
            type: String,
            required: true,
        },
        isEmailVerified: {
            type: Boolean,
            default: false,
        },

        consentGiven: {
            type: Boolean,
            default: false,
        },
        consentDate: {
            type: Date,
            default: null,
        }
    },
    { timestamps: true },
);

const User = mongoose.model("User", userSchema);
export default User;
